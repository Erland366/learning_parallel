import contextlib
from typing import List
import torch
import torch.distributed as dist
from torch import nn

import process_group_manager as pgm

### Begin data parallel naive
class DataParallelNaive(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        # Whether to synchronize gradients during backward pass. Set to False when using gradient accumulation
        self.require_backward_grad_sync = True
        self.register_backward_hook(self._all_reduce_grads)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def register_backward_hook(self, hook):
        """Register a backward hook for all parameters of the model that requires gradients"""
        for p in self.module.parameters():
            if p.requires_grad:
                p.register_hook(hook)

    def _all_reduce_grads(self, grad):
        """Perform an all-reduce operation to synchronize gradients across multiple processess."""
        # No synchronization needed during gradient accumulation, except at the final accumulation step.
        if self.require_backward_grad_sync:
            dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=pgm.process_group_manager.dp_group)
            grad /= pgm.process_group_manager.dp_world_size
        return grad

