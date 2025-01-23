"""
torchrun --nproc_per_node 2 train.py --tp_size 2 --run_name process_group_manager --use_wandb --micro_batch_size 4 --gradient_accumulation_steps 8 --max_tokens 4096 --num_proc 16 --run_name dataloader 
"""
from __future__ import annotations
import os
import time
import wandb
import datetime
import torch
import torch.nn.functional as F
import torch.distributed as dist
import argparse
from torch.optim import AdamW
from transformers import AutoConfig

import lovely_tensors as lt; lt.monkey_patch()

from dataloader import MicroBatchDataLoader
from model import Llama
import process_group_manager as pgm
from process_group_manager import setup_process_group_manager
from utils import set_all_seed, print, to_readable_format

def train_step(model: str, dataloader: "torch.utils.data.DataLoader", device):
    acc_loss = 0.0

    for i in range(dataloader.grad_acc_steps):
        batch = next(dataloader)
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)

        outputs = model(input_ids=input_ids) # B, S, V

        batch_size, seq_len = input_ids.shape
        target_ids = target_ids.reshape(-1) # (B * S)
        outputs = outputs.view(seq_len * batch_size, -1) # (B * S, V)

        # This means we are considering all loss even for our prefiling phase token
        loss = F.cross_entropy(outputs, target_ids, reduction="mean") / dataloader.grad_acc_steps

        loss.backward()

        acc_loss += loss.item()

    return acc_loss

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Training script for LLaMA model")
    
    # Environment arguments
    parser.add_argument("--omp_num_threads", type=str, default="1")
    parser.add_argument("--tokenizers_parallelism", type=str, default="false")

    # Dataset arguments
    parser.add_argument("--dataset_name", type=str, default="roneneldan/TinyStories")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--num_proc", type=int, default=4)

    # Model arguments
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM-360M-Instruct")
    parser.add_argument("--num_hidden_layers", type=int, default=2)
    parser.add_argument("--num_attention_heads", type=int, default=16)
    parser.add_argument("--num_key_value_heads", type=int, default=4)
    parser.add_argument("--hidden_size", type=int, default=512)

    # Training arguments
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=1e6)

    
    # Distributed training arguments
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor Parallel size")
    parser.add_argument("--dp_size", type=int, default=1, help="Data Parallel size")
    parser.add_argument("--pp_size", type=int, default=1, help="Pipeline Parallel size")
    parser.add_argument("--pp_engine", type=str, default="afab", choices=["1f1b", "afab"])

    # Logging arguments
    parser.add_argument("--run_name", type=str, default="default_run")
    parser.add_argument("--use_wandb", action="store_true")

    args = parser.parse_args()

    # Set environment variables
    os.environ["OMP_NUM_THREADS"] = args.omp_num_threads
    os.environ["TOKENIZERS_PARALLELISM"] = args.tokenizers_parallelism
    os.environ["DEVICE"] = "cuda"
    
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    backend = "nccl"
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dtype = torch.float32 # torch.cuda.is_bf16_supported() is broken on Kaggle, always use fp32 instead

    dist.init_process_group(rank=global_rank, world_size=world_size, backend=backend, init_method=f"env://", timeout=datetime.timedelta(minutes=2.0))
    setup_process_group_manager(args.dp_size, args.pp_size, args.tp_size)

    is_wandb_rank = pgm.process_group_manager.dp_rank == 0 and pgm.process_group_manager.tp_rank == 0 and pgm.process_group_manager.pp_is_last_stage
    set_all_seed(args.seed)

    if is_wandb_rank and args.use_wandb:
        wandb.init(
            project="picotron_tutorial",
            name=f"{args.run_name}_{pgm.process_group_manager}",
            config={
                "tensor_parallel_size": pgm.process_group_manager.tp_world_size,
                "pipeline_parallel_size": pgm.process_group_manager.pp_world_size,
                "data_parallel_size": pgm.process_group_manager.dp_world_size,
                "model": args.model_name,
                "learning_rate": args.learning_rate,
                "seed": args.seed,
            },
        )

    model_config = AutoConfig.from_pretrained(args.model_name)
    model_config.num_hidden_layers = args.num_hidden_layers
    model_config.num_attention_heads = args.num_attention_heads
    model_config.num_key_value_heads = args.num_key_value_heads
    model_config.max_position_embeddings = args.seq_len
    model_config.hidden_size = args.hidden_size
    if not hasattr(model_config, "head_dim"):
        model_config.head_dim = args.hidden_size // args.num_attention_heads
    if not hasattr(model_config, "num_key_values") and hasattr(model_config, "num_key_value_heads"):
        model_config.num_key_values = model_config.num_key_value_heads
    if not hasattr(model_config, "hidden_dim") and hasattr(model_config, "hidden_size"):
        model_config.hidden_dim = model_config.hidden_size

    model = Llama(config=model_config)
    model.to(dtype).to(device)            
    model.train()

    dist.barrier()

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    dist.barrier()
    
    # Create dummy data
    dataloader = MicroBatchDataLoader(
        seq_len=args.seq_len,
        micro_batch_size=args.micro_batch_size,
        grad_acc_steps=args.gradient_accumulation_steps,
        dataset_name=args.dataset_name,
        tokenizer_name=args.model_name,
        max_tokens=args.max_tokens,
        num_workers=args.num_workers,
        num_proc=args.num_proc,
    )

    # So this is assuming a pretrain phase where all the data sequence lengths 
    # are the same
    tokens_per_step = dataloader.global_batch_size * args.seq_len 
    if pgm.process_group_manager.global_rank == 0:
        print(f"Total tokens is : {to_readable_format(tokens_per_step)}", is_print_rank=is_wandb_rank)


    trained_token, step = 0, 0

    dist.barrier()

    while trained_token < args.max_tokens:
        step_start_time = time.time()

        optimizer.zero_grad()

        loss = train_step(model, dataloader=dataloader, device=device)

        optimizer.step()

        step_duration = time.time() - step_start_time
        trained_token += tokens_per_step
        step += 1

        print(
            f"[rank {pgm.process_group_manager.global_rank}] Step: {step}, Loss: {loss:.4f}",
            f"Global batch size (with seq_len) : {to_readable_format(tokens_per_step)}",
            f"Tokens/s : {to_readable_format(int(tokens_per_step / step_duration))}",
            f"Tokens/s/GPU : {to_readable_format(int(tokens_per_step / step_duration / world_size))}",
            f"Tokens: {to_readable_format(trained_token)} / {to_readable_format(args.max_tokens)}",
            f"Memory usage: {torch.cuda.memory_reserved() / 1e9:.2f}GB"
            , is_print_rank=is_wandb_rank
        )

        if is_wandb_rank and args.use_wandb:
            wandb.log(
                {
                    "loss" : loss,
                    "tokens_per_step" : tokens_per_step,
                    "tokens_per_second" : tokens_per_step / step_duration,
                    "memory_usage" : torch.cuda.memory_reserved() / 1e9,
                    "trained_tokens" : trained_token
                }
            )

    if is_wandb_rank and args.use_wandb:
        wandb.finish()

    dist.destroy_process_group()
