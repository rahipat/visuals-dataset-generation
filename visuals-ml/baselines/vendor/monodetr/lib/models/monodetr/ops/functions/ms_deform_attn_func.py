# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable

# VISUALS-MOD: the compiled CUDA op does not build on Windows. Fall back to the
# pure-PyTorch core (defined below) when it is unavailable, so the harness runs
# locally for smoke/overfit tests. HiPerGator compiles the op and uses the fast path.
try:
    import MultiScaleDeformableAttention as MSDA
    _HAS_CUDA_MSDA = True
except Exception:  # pragma: no cover - depends on local build
    MSDA = None
    _HAS_CUDA_MSDA = False


class MSDeformAttnFunction(Function):
    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):
        ctx.im2col_step = im2col_step
        # VISUALS-MOD: the compiled CUDA kernel only instantiates for float32/
        # float64 (see AT_DISPATCH_FLOATING_TYPES in ms_deform_attn_cuda.cu),
        # so it raises "not implemented for 'Half'" when called under AMP --
        # value/sampling_locations/attention_weights arrive as fp16 here
        # because they're produced by nn.Linear layers running inside the
        # autocast region. Rather than patch+recompile the CUDA kernel for
        # Half, force fp32 at this Function boundary and cast back afterward;
        # this is torch's documented pattern for custom ops that AMP doesn't
        # know how to autocast (see "Functions with multiple inputs or
        # autocastable ops" in the torch.cuda.amp notes).
        input_dtype = value.dtype
        value = value.float()
        sampling_locations = sampling_locations.float()
        attention_weights = attention_weights.float()
        output = MSDA.ms_deform_attn_forward(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights)
        ctx.input_dtype = input_dtype
        return output.to(input_dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = \
            MSDA.ms_deform_attn_backward(
                value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights,
                grad_output.float(), ctx.im2col_step)

        input_dtype = ctx.input_dtype
        return (grad_value.to(input_dtype), None, None,
                grad_sampling_loc.to(input_dtype), grad_attn_weight.to(input_dtype), None)


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    # for debug and test only,
    # need to use cuda version instead
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_*M_, D_, H_, W_)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                          mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_*M_, 1, Lq_, L_*P_)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_*D_, Lq_)
    return output.transpose(1, 2).contiguous()


# VISUALS-MOD: when the CUDA op is unavailable, route .apply() to the pure-PyTorch
# core (autograd-differentiable on its own, so no custom backward is needed).
class _MSDeformAttnPyTorch:
    @staticmethod
    def apply(value, value_spatial_shapes, value_level_start_index,
              sampling_locations, attention_weights, im2col_step):
        return ms_deform_attn_core_pytorch(
            value, value_spatial_shapes, sampling_locations, attention_weights)


if not _HAS_CUDA_MSDA:
    MSDeformAttnFunction = _MSDeformAttnPyTorch
