import pytest
import torch

from kernel import _layer_norm_fwd

def test_layer_norm():
    atol = 5e-2
    torch.random.manual_seed(0)
    batch_size = 8
    seqlen = 512
    hidden_size = 128
    device="cuda"
    dtype = torch.bfloat16
    args_tensor = dict(device=device, dtype=dtype, requires_grad=True)

    has_residual = True
    is_rms_norm = False
    has_x1 = True
    has_weight1 = False

    all_close = (
        lambda x, x_pt, x_ref, atol=atol: (x - x_ref).abs().max()
        <= 2 * (x_pt[~x_pt.isnan()] - x_ref[~x_pt.isnan()]).abs().max() + atol
        or (
            (x_pt[~x_pt.isnan()] - x_ref[~x_pt.isnan()]).abs().max() == 0.0
            and (x - x_ref).abs().max()
            <= 2 * (x_pt[~x_pt.isnan()] * 0.3 / 0.3 - x_ref[~x_pt.isnan()]).abs().max() + atol
        )
    )
    x0 = torch.randn(
        batch_size, seqlen, hidden_size, **args_tensor
    )
    x0_pt = x0.detach().clone().requires_grad_()
    x0_ref = x0.detach().clone().requires_grad_()
    if has_residual:
        res = torch.randn_like(x0, **args_tensor)
        res_pt = res.detach().clone().requires_grad_()
        res_ref = res.detach().clone().requires_grad_()
    else:
        res, res_pt, res_ref = None, None, None
    weight = torch.randn(hidden_size, **args_tensor)
    if not is_rms_norm:
        bias = torch.randn(hidden_size, **args_tensor)
    else:
        bias = None
    weight_pt = res.detach().clone().requires_grad_()
    weight_ref = res.detach().clone().requires_grad_()

    bias_pt = res.detach().clone().requires_grad_() if bias is not None else None
    bias_ref = res.detach().clone().requires_grad_() if bias is not None else None

    if has_x1:
        x1 = torch.rand_like(x0, **args_tensor)
        x1_pt = x1.detach().clone().requires_grad_()
        x1_ref = x1.detach().clone().requires_grad_()
    else:
        x1, x1_pt, x1_ref = None, None, None

    if has_weight1:
        weight1 = torch.rand_like(weight, **args_tensor)
        weight1_pt = weight1.detach().clone().requires_grad_()
        weight1_ref = weight1.detach().clone().requires_grad_()
        if not is_rms_norm:
            bias1 = torch.randn_like(bias, **args_tensor)
        else:
            bias1 = None
        bias1_pt = bias1.detach().clone().requires_grad_() if bias1 is not None else None
        bias1_ref = bias1.detach().clone().requires_grad_() if bias1 is not None else None
        
    else:
        weight1, weight1_pt, weight1_ref = None, None, None
        bias1, bias1_pt, bias1_ref = None, None, None

    rowscale = torch.randn(batch_size, seqlen, **args_tensor)
    
