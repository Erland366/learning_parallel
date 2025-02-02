import torch
from einops import rearrange, einsum
from torch import nn
from torch import functional as F

class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, tp_size, rank):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features // tp_size
        self.tp_size = tp_size
        self.rank = rank
        self.layer = nn.Linear(self.in_features, self.out_features)

    def forward(self, x):
        # Splitted into [B, S, D // tp_size]
        return self.layer(x)

class RowParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, tp_size, rank):
        super().__init__()
        self.in_features = in_features // tp_size
        self.out_features = out_features
        self.tp_size = tp_size
        self.rank = rank
        self.layer = nn.Linear(self.in_features, self.out_features)

    def forward(self, x):
        return self.layer(x)


class ParallelAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dtype=torch.bfloat16, device="cuda", tp_size: int = 2):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.dtype = dtype
        self.device = device
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.tp_size = tp_size
        self.head_dim_per_tp = self.head_dim // self.tp_size
        
        # Project on full hidden_dim instead of head_dim
        self.q_proj_1 = ColumnParallelLinear(hidden_dim, hidden_dim, tp_size=tp_size, rank=0).to(dtype).to(device)
        self.q_proj_2 = ColumnParallelLinear(hidden_dim, hidden_dim, tp_size=tp_size, rank=1).to(dtype).to(device)
        self.k_proj_1 = ColumnParallelLinear(hidden_dim, hidden_dim, tp_size=tp_size, rank=0).to(dtype).to(device)
        self.k_proj_2 = ColumnParallelLinear(hidden_dim, hidden_dim, tp_size=tp_size, rank=1).to(dtype).to(device)
        self.v_proj_1 = ColumnParallelLinear(hidden_dim, hidden_dim, tp_size=tp_size, rank=0).to(dtype).to(device)
        self.v_proj_2 = ColumnParallelLinear(hidden_dim, hidden_dim, tp_size=tp_size, rank=1).to(dtype).to(device)
        self.o_proj_1 = RowParallelLinear(hidden_dim, hidden_dim, tp_size=tp_size, rank=0).to(dtype).to(device)
        self.o_proj_2 = RowParallelLinear(hidden_dim, hidden_dim, tp_size=tp_size, rank=1).to(dtype).to(device)
        
        # Initialize scale factor
        self.scale = (self.head_dim) ** -0.5

    def forward(self, x):
        # [B, S, D]
        # Project first, then rearrange
        q_1 = rearrange(self.q_proj_1(x), "b s (h d) -> b h s d", h=self.num_heads, d=self.head_dim_per_tp) # [B, H, S, D/TP]
        k_1 = rearrange(self.k_proj_1(x), "b s (h d) -> b h s d", h=self.num_heads, d=self.head_dim_per_tp)
        v_1 = rearrange(self.v_proj_1(x), "b s (h d) -> b h s d", h=self.num_heads, d=self.head_dim_per_tp)
        q_2 = rearrange(self.q_proj_2(x), "b s (h d) -> b h s d", h=self.num_heads, d=self.head_dim_per_tp)
        k_2 = rearrange(self.k_proj_2(x), "b s (h d) -> b h s d", h=self.num_heads, d=self.head_dim_per_tp)
        v_2 = rearrange(self.v_proj_2(x), "b s (h d) -> b h s d", h=self.num_heads, d=self.head_dim_per_tp)
        
        attention_scores_1 = einsum(q_1, k_1, "b h s1 d, b h s2 d -> b h s1 s2") # [B, H, S1, S2]
        attention_scores_2 = einsum(q_1, k_1, "b h s1 d, b h s2 d -> b h s1 s2")

        attention_probs_1 = F.softmax(attention_scores_1, dim=-1) / self.scale
        attention_probs_2 = F.softmax(attention_scores_1, dim=-1) / self.scale

        out_1 = einsum(attention_probs_1, v_1, "b h s2 s1, b h s1 d -> b h s2 d") # [B, H, S, D/TP]
        out_2 = einsum(attention_probs_2, v_2, "b h s2 s1, b h s1 d -> b h s2 d") # [B, H, S, D/TP]

        out_1 = rearrange(out_1, "b h s2 d -> b s2 (h d)")
        out_2 = rearrange(out_2, "b h s2 d -> b s2 (h d)")

        out_1 = self.o_proj_1(out_1)
        out_2 = self.o_proj_2(out_2)
        
        return torch.concat((out_1, out_2), dim=-1)


B = 2
S = 16
D = 128
example_input = torch.randn((B, S, D), device="cuda", dtype=torch.bfloat16)
parallel_attention = ParallelAttention(D, 2)
output = parallel_attention(example_input)
print(output)
