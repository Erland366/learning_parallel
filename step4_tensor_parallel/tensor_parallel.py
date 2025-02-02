import math
from typing import Optional
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import process_group_manager as pgm

### begin TP communications
def split_tensor_along_last_dim(tensor, num_partitions):
    """Split a tensor along its last dimension into num_partitions chunks."""
    last_dim = tensor.dim() - 1
    assert tensor.size()[last_dim] % num_partitions == 0, f"{tensor.size()[last_dim]} is not divisible by {num_partitions}"
    last_dim_size = tensor.size()[last_dim] // num_partitions
    return torch.split(tensor, last_dim_size, dim=last_dim)

class Reduce(torch.autograd.Function):
    """All-reduce in forward pass, identity in backward pass."""
    @staticmethod
    def forward(ctx, input):
        if pgm.process_group_manager.tp_world_size == 1:
            return input
        dist.all_reduce(input, op=dist.ReduceOp.SUM, group=pgm.process_group_manager.tp_group)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class Gather(torch.autograd.Function):
    """Gather in forward pass, split in backward pass."""
    @staticmethod
    def forward(ctx, input):
        if pgm.process_group_manager.tp_world_size == 1:
            return input
        last_dim = input.dim() - 1
        # Need contiguous tensors for collectives -> https://github.com/pytorch/pytorch/blob/main/torch/distributed/nn/functional.py#L321
        input = input.contiguous()
        tensor_list = [torch.empty_like(input) for _ in range(pgm.process_group_manager.tp_world_size)]
        tensor_list[pgm.process_group_manager.tp_rank] = input
        dist.all_gather(tensor_list, input, group=pgm.process_group_manager.tp_group)
        output = torch.cat(tensor_list, dim=last_dim).contiguous()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if pgm.process_group_manager.tp_world_size == 1:
            return grad_output
        # Split gradient according to TP size
        chunks = split_tensor_along_last_dim(grad_output, pgm.process_group_manager.tp_world_size)
        return chunks[pgm.process_group_manager.tp_rank].contiguous()

class Copy(torch.autograd.Function):
    """Identity in forward pass, all-reduce in backward pass."""
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        if pgm.process_group_manager.tp_world_size == 1:
          return grad_output
        dist.all_reduce(grad_output, op=dist.ReduceOp.SUM, group=pgm.process_group_manager.tp_group)
        return grad_output

### end TP communications

def apply_tensor_parallel(model):
    def _replace_module(_module, _linear_proj_name, _style, args={}):
        assert _style in ["column", "row", "vocab"]

        linear_layer = getattr(_module, _linear_proj_name)
        if _style == "column":
            new_linear_layer = ColumnParallelLinear(
                in_features=linear_layer.in_features,
                out_features=linear_layer.out_features,
                bias=linear_layer.bias is not None,
                gather_output=args.get("gather_output", False)
            )
        elif _style == "row":
            new_linear_layer = RowParallelLinear(
                in_features=linear_layer.in_features,
                out_features=linear_layer.out_features,
                bias=linear_layer.bias is not None,
            )
        else:
            new_linear_layer = VocabParallelEmbedding(
                num_embeddings=linear_layer.num_embeddings,
                embedding_dim=linear_layer.embedding_dim,
            )
        setattr(_module, _linear_proj_name, new_linear_layer)

    module_linear_name_stype_mapping_list = [
        ("attention", "q_proj", "column"),
        ("attention", "k_proj", "column"),
        ("attention", "v_proj", "column"),
        ("attention", "out_proj", "row"),
        ("mlp", "up_proj", "column"),
        ("mlp", "gate_proj", "column"),
        ("mlp", "down_proj", "row")
    ]

    for layer in model.decoder_layers:
        for module_name, linear_proj_name, style in module_linear_name_stype_mapping_list:
            _replace_module(getattr(layer, module_name), linear_proj_name, style)

    _replace_module(model, "embedding", "vocab")
    _replace_module(model, "final_proj", "column", args={"gather_output": True})

    return model

class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool, gather_output: bool = False):
        super().__init__()

        self.tp_world_size = pgm.process_group_manager.tp_world_size
        self.tp_rank = pgm.process_group_manager.tp_rank

        self.in_features = in_features
        self.out_features = out_features
        assert out_features % self.tp_world_size == 0, "Hidden dimension must be divisible by the tensor parallel world size"
        self.output_size_per_partition = self.out_features // self.tp_world_size
        self.gather_output = gather_output

        self.weight = nn.Parameter(torch.Tensor(self.output_size_per_partition, self.in_features)) # W_i
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_size_per_partition))
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.tp_world_size == 1:
            # U(-sqrt(k), sqrt(k))
            k = 1 / self.weight.size(1) # it's based on the input dimension
            bound = math.sqrt(k)
            torch.nn.init.uniform_(self.weight, -bound, bound)
            return

        # When TP > 1, Initialize master weights
        # We don't need to put the master weight on the GPU
        master_weight = torch.empty(self.out_features, self.in_features, dtype=self.weight.dtype, requires_grad=False)

        # Initialized based on the master weight
        k = 1 / self.weight.size(1) # it's based on the input dimension
        bound = math.sqrt(k)
        torch.nn.init.uniform_(master_weight, -bound, bound)

        weight_list = torch.split(master_weight, split_size_or_sections=self.output_size_per_partition, dim=0)
        self.weight.data = weight_list[self.tp_rank].contiguous()

    def forward(self, x):
        input_parallel = Copy.apply(x)
        # XW_i^T + b, output is y_i

        output = F.linear(input_parallel, self.weight, self.bias)
        if self.gather_output:
            output = Gather.apply(output)
        return output


class RowParallelLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool):
        super().__init__()

        self.tp_world_size = pgm.process_group_manager.tp_world_size
        self.tp_rank = pgm.process_group_manager.tp_rank

        self.in_features = in_features
        self.out_features = out_features
        assert self.in_features % self.tp_world_size == 0, "In features must be a multiple of tp_world_size"
        self.in_feature_per_partition = self.in_features // self.tp_world_size
        self.weight = nn.Parameter(torch.Tensor(self.out_features, self.in_feature_per_partition))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_features))
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        if pgm.process_group_manager.tp_world_size == 1:
            # Apply initialization by random uniform of k
            k = 1 / self.weight.size(1)
            bound = math.sqrt(k)
            torch.nn.init.uniform_(self.weight, -bound, bound)
            return
        # Create full bound first
        # Maybe we don't need to put this master weight on GPU?
        master_weights = torch.empty(self.out_features, self.in_features, dtype=self.weight.dtype, requires_grad=False)

        k = 1 / master_weights.size(1)
        bound = math.sqrt(k)
        torch.nn.init.uniform_(master_weights, -bound, bound)

        weight_list = torch.split(master_weights, self.in_feature_per_partition, dim=1)
        self.weight.data = weight_list[self.tp_rank].contiguous()

    def forward(self, x):
        output = F.linear(x, self.weight)

        output = Reduce.apply(output)

        return output if self.bias is None else output + self.bias

class VocabParallelEmbedding(nn.Module):
    def __init__(
        self, 
        num_embeddings: int, 
        embedding_dim: int, 
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scaled_grad_by_freq: bool = False,
        sparse: bool = False
    ):
        super().__init__()

        self.tp_world_size = pgm.process_group_manager.tp_world_size
        self.tp_rank = pgm.process_group_manager.tp_rank

        self.num_embeddings = num_embeddings
        self.sparse = sparse
        self.scaled_grad_by_freq = scaled_grad_by_freq
        self.norm_type = norm_type
        self.max_norm = max_norm
        self.padding_idx = padding_idx
        self.embedding_dim = embedding_dim

        self.vocab_start_idx, self.vocab_end_idx = self._vocab_range_from_global_vocab_size(
            self.num_embeddings, pgm.process_group_manager.tp_rank, pgm.process_group_manager.tp_world_size
        )
        self.num_embeddings_per_partition = self.vocab_end_idx - self.vocab_start_idx

        self.weight = nn.Parameter(torch.Tensor(self.num_embeddings_per_partition, self.embedding_dim))

        self.reset_parameters()

    def _vocab_range_from_global_vocab_size(self, global_vocab_size: int, rank: int, world_size: int):
        assert global_vocab_size % world_size == 0, f"{global_vocab_size} is not divisible by {world_size}" 
        per_partition_vocab_size = global_vocab_size // world_size
        index_f = rank * per_partition_vocab_size
        index_l = index_f + per_partition_vocab_size
        return index_f, index_l

    def reset_parameters(self):
        if self.tp_world_size == 1:
            # Initialize vocab embedding with N(0, 1)
            torch.nn.init.normal_(self.weight, mean=0.0, std=1.0)
            return

        master_weight = torch.empty(self.num_embeddings, self.embedding_dim, dtype=self.weight.dtype, requires_grad=False)
        torch.nn.init.normal_(master_weight, mean=0.0, std=1.0)

        weight_list = torch.split(master_weight, self.num_embeddings_per_partition, dim=0)
        self.weight.data = weight_list[self.tp_rank].contiguous()

    def forward(self, input):
        """
        Performs an embedding lookup for input tokens in the parallelized embedding layer
        1. Masks tokens that fall outside the specified vocabulary range and adjusts the input
        2. Performs embedding lookups for valid tokens, setting embeddings of out-of-vocabulary tokens to zero
        3. Reduces the embeddings across model parallel GPUs using all-reduce for synchronization
        """
        # Build the mask for out-of-vocabulary tokens.
        input_mask = (input < self.vocab_start_idx) | (input >= self.vocab_end_idx)
        # Mask the input.
        masked_input = input.clone() - self.vocab_start_idx
        masked_input[input_mask] = 0
        # Get the embeddings for the valid tokens.
        output_parallel = F.embedding(
            masked_input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scaled_grad_by_freq,
            self.sparse,
        )
        # Embedding of out-of-vocabulary tokens is set to 0.
        output_parallel[input_mask, :] = 0.0
        output = Reduce.apply(output_parallel)
        return output
