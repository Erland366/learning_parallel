import math
from typing import Optional
import torch
import torch.nn as nn
import torch.distribution as dist
import torch.nn.functional as F
import process_group_manager as pgm

## Begin TP Communications
def split_tensor_along_last_dim(tensor, num_partitions):
    """Split a tensor along its last dimension into num_partitions chunks."""
    last_dim = tensor.dim() - 1
    assert tensor.size()[last_dim] % num_partitions == 0, f"{tensor.size()[last_dim]} is not divisible by {num_partitions}"
    last_dim_size = tensor.size()[last_dim] // num_partitions
    return torch.split(tensor, last_dim_size, dim=last_dim)

class Reduce(torch.autograd.Function):
    """All-reduce in forward pass, identity in backward pass"""
    @staticmethod
    def forward(ctx, inp):
        if pgm.process_group_manager.tp_world_size == 1:
            return inp
        dist.all_reduce(inp, op=dist.ReduceOp.SUM, group=pgm.process_group_manager.tp_group)
        return inp

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs

class Gather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp):
        if pgm.process_group_manager.tp_world_size == 1:
            return inp
        last_dim = inp.dim() - 1
        # Need to be contiguous for collectives https://github.com/pytorch/pytorch/blob/main/torch/distributed/nn/functional.py#L321
        inp = inp.contiguous()
        tensor_list = [torch.empty_like(inp) for _ in range(pgm.process_group_manager.tp_world_size)]
        tensor_list[pgm.process_group_manager.tp_rank] = inp
        dist.all_gather(tensor_list, inp, group=pgm.process_group_manager.tp_group)
        output = torch.cat(tensor_list, dim=last_dim).contiguous()
        return output

