import torch
from torch.utils.data import DataLoader
import numpy as np
from functools import partial
from datasets import Features, Sequence,Value, load_dataset
from transformers import AutoTokenizer

import step3_dataloader.process_group_manager as pgm

class MicroBatchDataLoader(DataLoader):
    def __init__(self, seq_len, micro_batch_size, grad_acc_steps, dataset_name, tokenizer_name, max_tokens, num_workers, num_proc, split="train"):
        self.micro_batch_size = micro_batch_size
        self.grad_acc_steps = grad_acc_steps
        self.seq_len = seq_len

        self.global_batch_size = micro_batch_size * grad_acc_steps * pgm.process_group_manager.dp_world_size

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.dataset = load_dataset(dataset_name, split=split)

        self.tokenized_dataset = self.tokenizer(self.dataset, "text", max_length=self.seq_len)

        total_tokens = self.tokenized_dataset.num_row * (self.seq_len + 1)
        assert total_tokens >= max_tokens, f"Not enough tokens, Have {total_tokens} tokens but need {max_tokens} token instead"

        super().__init__(
            self.tokenized_dataset,
            batch_size=self.micro_batch_size,
            collate_fn=self.collate_batch,
            pin_memory=True,
            num_workers=num_workers,
            shuffle=False
        )

    def tokenizer_group_text(self, examples, tokenizer, sequence_length):
        """Tokenize a list of texts and group them in chunks of sequence_length + 1
        Seems like there's a confusion of the +1 at the end
        Seems like they designed to like, make it overlap with each other?
        """
        tokenized_text_batch = tokenizer.batch_encode_plus(
            examples,
            return_attention_mask=False,
            return_token_type_ids=False,
            return_tensors="np"
        )
        concatenated_text = {"input_ids" : np.concatenate(tokenized_text_batch["input_ids"])}
        total_length = len(concatenated_text["input_ids"])

        if total_length >= sequence_length + 1: # This is making sure that we cut off the total_length but at least keep 1 of the data
            total_length = ((total_length - 1) // sequence_length) * sequence_length + 1 

        result = {
            "input_ids" : [
                concatenated_text["input_ids"][i : i + sequence_length + 1] for i in range(0, total_length - sequence_length, sequence_length)
            ]
        } 

        return result

    def tokenize_dataset(self, dataset, text_column_name, sequence_length, num_proc):
        tokenizer_func = partial(
            self.tokenizer_group_text,
            tokenizer=self.tokenizer,
            sequence_length=sequence_length
        )

        tokenized_dataset = dataset.map(
            tokenizer_func,
            input_columns=text_column_name,
            remove_columns=dataset.column_names,
            features=Features({
                "input_ids" : Sequence(feature=Value(dtype="int64"), length=sequence_length + 1)
            }),
            bached=True,
            num_proc=num_proc,
            load_from_cache_file=True, # Preprocess dataset only once and cache it
            desc=f"Grouping texts in chunks of {sequence_length - 1}"
        )

        return tokenized_dataset

    def collate_batch(self, batch):
        batch_input_ids = torch.stack([torch.tensor(item["input_ids"]) for item in batch])
        batch_size = batch_input_ids.size(0)
        input_ids = batch_input_ids[:, :-1].contiguous()
        label_ids = batch_input_ids[:, 1:].contiguous()
        position_ids = torch.arange(0, self.seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1).contiguous()
        attn_mask = torch.tril(torch.ones((self.seq_len, self.seq_len), dtype=torch.bool))
        attn_mask = attn_mask.unsqueeze(0).expand(batch_size, -1, -1).contiguous()

        return {
            "input_ids" : input_ids,
            "label_ids" : label_ids,
            "position_ids" : position_ids,
            "attn_mask" : attn_mask
        }

    def __iter__(self):
        if self._iterator is None:
            self._iterator = super().__iter__()
        return self

    def __next__(self):
        if self._iterator is None:
            self._iterator = super().__iter__()

        try:
            batch = next(self._iterator)
        except StopIteration:
            self._iterator = None
            raise StopIteration
        return batch
