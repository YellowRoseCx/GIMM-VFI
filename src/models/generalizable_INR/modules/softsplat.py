# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# softmax-splatting: https://github.com/sniklaus/softmax-splatting
# --------------------------------------------------------

import torch
import triton
import triton.language as tl

@triton.jit
def softsplat_fwd_kernel(
    tenIn_ptr, tenFlow_ptr, tenOut_ptr,
    N, C, H, W,
    stride_in_n, stride_in_c, stride_in_h, stride_in_w,
    stride_flow_n, stride_flow_c, stride_flow_h, stride_flow_w,
    stride_out_n, stride_out_c, stride_out_h, stride_out_w,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    intIndex = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = intIndex < N * C * H * W

    intX = intIndex % W
    intY = (intIndex // W) % H
    intC = (intIndex // (W * H)) % C
    intN = (intIndex // (W * H * C)) % N

    offset_flow_x = intN * stride_flow_n + 0 * stride_flow_c + intY * stride_flow_h + intX * stride_flow_w
    offset_flow_y = intN * stride_flow_n + 1 * stride_flow_c + intY * stride_flow_h + intX * stride_flow_w

    flowX = tl.load(tenFlow_ptr + offset_flow_x, mask=mask, other=0.0)
    flowY = tl.load(tenFlow_ptr + offset_flow_y, mask=mask, other=0.0)

    fltX = intX + flowX
    fltY = intY + flowY

    intNorthwestX = tl.math.floor(fltX)
    intNorthwestY = tl.math.floor(fltY)

    intNorthwestX_int = intNorthwestX.to(tl.int32)
    intNorthwestY_int = intNorthwestY.to(tl.int32)

    intNortheastX_int = intNorthwestX_int + 1
    intNortheastY_int = intNorthwestY_int

    intSouthwestX_int = intNorthwestX_int
    intSouthwestY_int = intNorthwestY_int + 1

    intSoutheastX_int = intNorthwestX_int + 1
    intSoutheastY_int = intNorthwestY_int + 1

    fltNorthwest = (intSoutheastX_int - fltX) * (intSoutheastY_int - fltY)
    fltNortheast = (fltX - intSouthwestX_int) * (intSouthwestY_int - fltY)
    fltSouthwest = (intNortheastX_int - fltX) * (fltY - intNortheastY_int)
    fltSoutheast = (fltX - intNorthwestX_int) * (fltY - intNorthwestY_int)

    offset_in = intN * stride_in_n + intC * stride_in_c + intY * stride_in_h + intX * stride_in_w
    fltIn = tl.load(tenIn_ptr + offset_in, mask=mask, other=0.0)

    maskNW = mask & (intNorthwestX_int >= 0) & (intNorthwestX_int < W) & (intNorthwestY_int >= 0) & (intNorthwestY_int < H)
    offsetNW = intN * stride_out_n + intC * stride_out_c + intNorthwestY_int * stride_out_h + intNorthwestX_int * stride_out_w
    tl.atomic_add(tenOut_ptr + offsetNW, fltIn * fltNorthwest, mask=maskNW)

    maskNE = mask & (intNortheastX_int >= 0) & (intNortheastX_int < W) & (intNortheastY_int >= 0) & (intNortheastY_int < H)
    offsetNE = intN * stride_out_n + intC * stride_out_c + intNortheastY_int * stride_out_h + intNortheastX_int * stride_out_w
    tl.atomic_add(tenOut_ptr + offsetNE, fltIn * fltNortheast, mask=maskNE)

    maskSW = mask & (intSouthwestX_int >= 0) & (intSouthwestX_int < W) & (intSouthwestY_int >= 0) & (intSouthwestY_int < H)
    offsetSW = intN * stride_out_n + intC * stride_out_c + intSouthwestY_int * stride_out_h + intSouthwestX_int * stride_out_w
    tl.atomic_add(tenOut_ptr + offsetSW, fltIn * fltSouthwest, mask=maskSW)

    maskSE = mask & (intSoutheastX_int >= 0) & (intSoutheastX_int < W) & (intSoutheastY_int >= 0) & (intSoutheastY_int < H)
    offsetSE = intN * stride_out_n + intC * stride_out_c + intSoutheastY_int * stride_out_h + intSoutheastX_int * stride_out_w
    tl.atomic_add(tenOut_ptr + offsetSE, fltIn * fltSoutheast, mask=maskSE)

@triton.jit
def softsplat_bwd_ingrad_kernel(
    tenIn_ptr, tenFlow_ptr, tenOutgrad_ptr, tenIngrad_ptr,
    N, C, H, W,
    stride_in_n, stride_in_c, stride_in_h, stride_in_w,
    stride_flow_n, stride_flow_c, stride_flow_h, stride_flow_w,
    stride_outg_n, stride_outg_c, stride_outg_h, stride_outg_w,
    stride_ing_n, stride_ing_c, stride_ing_h, stride_ing_w,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    intIndex = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = intIndex < N * C * H * W

    intX = intIndex % W
    intY = (intIndex // W) % H
    intC = (intIndex // (W * H)) % C
    intN = (intIndex // (W * H * C)) % N

    offset_flow_x = intN * stride_flow_n + 0 * stride_flow_c + intY * stride_flow_h + intX * stride_flow_w
    offset_flow_y = intN * stride_flow_n + 1 * stride_flow_c + intY * stride_flow_h + intX * stride_flow_w

    flowX = tl.load(tenFlow_ptr + offset_flow_x, mask=mask, other=0.0)
    flowY = tl.load(tenFlow_ptr + offset_flow_y, mask=mask, other=0.0)

    fltX = intX + flowX
    fltY = intY + flowY

    intNorthwestX = tl.math.floor(fltX)
    intNorthwestY = tl.math.floor(fltY)

    intNorthwestX_int = intNorthwestX.to(tl.int32)
    intNorthwestY_int = intNorthwestY.to(tl.int32)

    intNortheastX_int = intNorthwestX_int + 1
    intNortheastY_int = intNorthwestY_int

    intSouthwestX_int = intNorthwestX_int
    intSouthwestY_int = intNorthwestY_int + 1

    intSoutheastX_int = intNorthwestX_int + 1
    intSoutheastY_int = intNorthwestY_int + 1

    fltNorthwest = (intSoutheastX_int - fltX) * (intSoutheastY_int - fltY)
    fltNortheast = (fltX - intSouthwestX_int) * (intSouthwestY_int - fltY)
    fltSouthwest = (intNortheastX_int - fltX) * (fltY - intNortheastY_int)
    fltSoutheast = (fltX - intNorthwestX_int) * (fltY - intNorthwestY_int)

    fltIngrad = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    maskNW = mask & (intNorthwestX_int >= 0) & (intNorthwestX_int < W) & (intNorthwestY_int >= 0) & (intNorthwestY_int < H)
    offsetNW = intN * stride_outg_n + intC * stride_outg_c + intNorthwestY_int * stride_outg_h + intNorthwestX_int * stride_outg_w
    outgNW = tl.load(tenOutgrad_ptr + offsetNW, mask=maskNW, other=0.0)
    fltIngrad += outgNW * fltNorthwest

    maskNE = mask & (intNortheastX_int >= 0) & (intNortheastX_int < W) & (intNortheastY_int >= 0) & (intNortheastY_int < H)
    offsetNE = intN * stride_outg_n + intC * stride_outg_c + intNortheastY_int * stride_outg_h + intNortheastX_int * stride_outg_w
    outgNE = tl.load(tenOutgrad_ptr + offsetNE, mask=maskNE, other=0.0)
    fltIngrad += outgNE * fltNortheast

    maskSW = mask & (intSouthwestX_int >= 0) & (intSouthwestX_int < W) & (intSouthwestY_int >= 0) & (intSouthwestY_int < H)
    offsetSW = intN * stride_outg_n + intC * stride_outg_c + intSouthwestY_int * stride_outg_h + intSouthwestX_int * stride_outg_w
    outgSW = tl.load(tenOutgrad_ptr + offsetSW, mask=maskSW, other=0.0)
    fltIngrad += outgSW * fltSouthwest

    maskSE = mask & (intSoutheastX_int >= 0) & (intSoutheastX_int < W) & (intSoutheastY_int >= 0) & (intSoutheastY_int < H)
    offsetSE = intN * stride_outg_n + intC * stride_outg_c + intSoutheastY_int * stride_outg_h + intSoutheastX_int * stride_outg_w
    outgSE = tl.load(tenOutgrad_ptr + offsetSE, mask=maskSE, other=0.0)
    fltIngrad += outgSE * fltSoutheast

    offset_ing = intN * stride_ing_n + intC * stride_ing_c + intY * stride_ing_h + intX * stride_ing_w
    tl.store(tenIngrad_ptr + offset_ing, fltIngrad, mask=mask)

@triton.jit
def softsplat_bwd_flowgrad_kernel(
    tenIn_ptr, tenFlow_ptr, tenOutgrad_ptr, tenFlowgrad_ptr,
    N, C_in, H, W,
    stride_in_n, stride_in_c, stride_in_h, stride_in_w,
    stride_flow_n, stride_flow_c, stride_flow_h, stride_flow_w,
    stride_outg_n, stride_outg_c, stride_outg_h, stride_outg_w,
    stride_flowg_n, stride_flowg_c, stride_flowg_h, stride_flowg_w,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    intIndex = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = intIndex < N * 2 * H * W

    intX = intIndex % W
    intY = (intIndex // W) % H
    intC_flow = (intIndex // (W * H)) % 2
    intN = (intIndex // (W * H * 2)) % N

    offset_flow_x = intN * stride_flow_n + 0 * stride_flow_c + intY * stride_flow_h + intX * stride_flow_w
    offset_flow_y = intN * stride_flow_n + 1 * stride_flow_c + intY * stride_flow_h + intX * stride_flow_w

    flowX = tl.load(tenFlow_ptr + offset_flow_x, mask=mask, other=0.0)
    flowY = tl.load(tenFlow_ptr + offset_flow_y, mask=mask, other=0.0)

    fltX = intX + flowX
    fltY = intY + flowY

    intNorthwestX = tl.math.floor(fltX)
    intNorthwestY = tl.math.floor(fltY)

    intNorthwestX_int = intNorthwestX.to(tl.int32)
    intNorthwestY_int = intNorthwestY.to(tl.int32)

    intNortheastX_int = intNorthwestX_int + 1
    intNortheastY_int = intNorthwestY_int

    intSouthwestX_int = intNorthwestX_int
    intSouthwestY_int = intNorthwestY_int + 1

    intSoutheastX_int = intNorthwestX_int + 1
    intSoutheastY_int = intNorthwestY_int + 1

    fltNorthwest = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    fltNortheast = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    fltSouthwest = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    fltSoutheast = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    mask_c0 = mask & (intC_flow == 0)
    mask_c1 = mask & (intC_flow == 1)

    fltNorthwest = tl.where(mask_c0, -1.0 * (intSoutheastY_int - fltY), fltNorthwest)
    fltNortheast = tl.where(mask_c0,  1.0 * (intSouthwestY_int - fltY), fltNortheast)
    fltSouthwest = tl.where(mask_c0, -1.0 * (fltY - intNortheastY_int), fltSouthwest)
    fltSoutheast = tl.where(mask_c0,  1.0 * (fltY - intNorthwestY_int), fltSoutheast)

    fltNorthwest = tl.where(mask_c1, (intSoutheastX_int - fltX) * -1.0, fltNorthwest)
    fltNortheast = tl.where(mask_c1, (fltX - intSouthwestX_int) * -1.0, fltNortheast)
    fltSouthwest = tl.where(mask_c1, (intNortheastX_int - fltX) *  1.0, fltSouthwest)
    fltSoutheast = tl.where(mask_c1, (fltX - intNorthwestX_int) *  1.0, fltSoutheast)

    fltFlowgrad = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for intChannel in range(C_in):
        offset_in = intN * stride_in_n + intChannel * stride_in_c + intY * stride_in_h + intX * stride_in_w
        fltIn = tl.load(tenIn_ptr + offset_in, mask=mask, other=0.0)

        maskNW = mask & (intNorthwestX_int >= 0) & (intNorthwestX_int < W) & (intNorthwestY_int >= 0) & (intNorthwestY_int < H)
        offsetNW = intN * stride_outg_n + intChannel * stride_outg_c + intNorthwestY_int * stride_outg_h + intNorthwestX_int * stride_outg_w
        outgNW = tl.load(tenOutgrad_ptr + offsetNW, mask=maskNW, other=0.0)
        fltFlowgrad += outgNW * fltIn * fltNorthwest

        maskNE = mask & (intNortheastX_int >= 0) & (intNortheastX_int < W) & (intNortheastY_int >= 0) & (intNortheastY_int < H)
        offsetNE = intN * stride_outg_n + intChannel * stride_outg_c + intNortheastY_int * stride_outg_h + intNortheastX_int * stride_outg_w
        outgNE = tl.load(tenOutgrad_ptr + offsetNE, mask=maskNE, other=0.0)
        fltFlowgrad += outgNE * fltIn * fltNortheast

        maskSW = mask & (intSouthwestX_int >= 0) & (intSouthwestX_int < W) & (intSouthwestY_int >= 0) & (intSouthwestY_int < H)
        offsetSW = intN * stride_outg_n + intChannel * stride_outg_c + intSouthwestY_int * stride_outg_h + intSouthwestX_int * stride_outg_w
        outgSW = tl.load(tenOutgrad_ptr + offsetSW, mask=maskSW, other=0.0)
        fltFlowgrad += outgSW * fltIn * fltSouthwest

        maskSE = mask & (intSoutheastX_int >= 0) & (intSoutheastX_int < W) & (intSoutheastY_int >= 0) & (intSoutheastY_int < H)
        offsetSE = intN * stride_outg_n + intChannel * stride_outg_c + intSoutheastY_int * stride_outg_h + intSoutheastX_int * stride_outg_w
        outgSE = tl.load(tenOutgrad_ptr + offsetSE, mask=maskSE, other=0.0)
        fltFlowgrad += outgSE * fltIn * fltSoutheast

    offset_flowg = intN * stride_flowg_n + intC_flow * stride_flowg_c + intY * stride_flowg_h + intX * stride_flowg_w
    tl.store(tenFlowgrad_ptr + offset_flowg, fltFlowgrad, mask=mask)

class softsplat_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tenIn, tenFlow):
        tenOut = tenIn.new_zeros([tenIn.shape[0], tenIn.shape[1], tenIn.shape[2], tenIn.shape[3]])

        N, C, H, W = tenIn.shape
        grid = lambda meta: (triton.cdiv(N * C * H * W, meta['BLOCK_SIZE']),)

        softsplat_fwd_kernel[grid](
            tenIn, tenFlow, tenOut,
            N, C, H, W,
            tenIn.stride(0), tenIn.stride(1), tenIn.stride(2), tenIn.stride(3),
            tenFlow.stride(0), tenFlow.stride(1), tenFlow.stride(2), tenFlow.stride(3),
            tenOut.stride(0), tenOut.stride(1), tenOut.stride(2), tenOut.stride(3),
            BLOCK_SIZE=512
        )

        ctx.save_for_backward(tenIn, tenFlow)
        return tenOut

    @staticmethod
    def backward(ctx, tenOutgrad):
        tenIn, tenFlow = ctx.saved_tensors

        # WE REMOVED .contiguous() HERE TO AVOID OVERHEAD
        # tenOutgrad = tenOutgrad.contiguous()

        N, C, H, W = tenIn.shape

        tenIngrad = None
        if ctx.needs_input_grad[0]:
            tenIngrad = tenIn.new_zeros([N, C, H, W])
            grid_in = lambda meta: (triton.cdiv(N * C * H * W, meta['BLOCK_SIZE']),)
            softsplat_bwd_ingrad_kernel[grid_in](
                tenIn, tenFlow, tenOutgrad, tenIngrad,
                N, C, H, W,
                tenIn.stride(0), tenIn.stride(1), tenIn.stride(2), tenIn.stride(3),
                tenFlow.stride(0), tenFlow.stride(1), tenFlow.stride(2), tenFlow.stride(3),
                tenOutgrad.stride(0), tenOutgrad.stride(1), tenOutgrad.stride(2), tenOutgrad.stride(3),
                tenIngrad.stride(0), tenIngrad.stride(1), tenIngrad.stride(2), tenIngrad.stride(3),
                BLOCK_SIZE=512
            )

        tenFlowgrad = None
        if ctx.needs_input_grad[1]:
            tenFlowgrad = tenFlow.new_zeros([N, 2, H, W])
            grid_flow = lambda meta: (triton.cdiv(N * 2 * H * W, meta['BLOCK_SIZE']),)
            softsplat_bwd_flowgrad_kernel[grid_flow](
                tenIn, tenFlow, tenOutgrad, tenFlowgrad,
                N, C, H, W,
                tenIn.stride(0), tenIn.stride(1), tenIn.stride(2), tenIn.stride(3),
                tenFlow.stride(0), tenFlow.stride(1), tenFlow.stride(2), tenFlow.stride(3),
                tenOutgrad.stride(0), tenOutgrad.stride(1), tenOutgrad.stride(2), tenOutgrad.stride(3),
                tenFlowgrad.stride(0), tenFlowgrad.stride(1), tenFlowgrad.stride(2), tenFlowgrad.stride(3),
                BLOCK_SIZE=512
            )

        return tenIngrad, tenFlowgrad

def softsplat(tenIn, tenFlow, tenMetric, strMode, return_norm=False):
    assert strMode.split("-")[0] in ["sum", "avg", "linear", "softmax"]

    if strMode == "sum":
        assert tenMetric is None
    if strMode == "avg":
        assert tenMetric is None
    if strMode.split("-")[0] == "linear":
        assert tenMetric is not None
    if strMode.split("-")[0] == "softmax":
        assert tenMetric is not None

    if strMode == "avg":
        tenIn = torch.cat(
            [
                tenIn,
                tenIn.new_ones([tenIn.shape[0], 1, tenIn.shape[2], tenIn.shape[3]]),
            ],
            1,
        )

    elif strMode.split("-")[0] == "linear":
        tenIn = torch.cat([tenIn * tenMetric, tenMetric], 1)

    elif strMode.split("-")[0] == "softmax":
        tenIn = torch.cat([tenIn * tenMetric.exp(), tenMetric.exp()], 1)

    if torch.isnan(tenIn).any():
        print("NaN values detected during training in tenIn. Exiting.")
        assert False

    tenOut = softsplat_func.apply(tenIn, tenFlow)

    if torch.isnan(tenOut).any():
        print("NaN values detected during training in tenOut_1. Exiting.")
        assert False

    if strMode.split("-")[0] in ["avg", "linear", "softmax"]:
        tenNormalize = tenOut[:, -1:, :, :]

        if len(strMode.split("-")) == 1:
            tenNormalize = tenNormalize + 0.0000001

        elif strMode.split("-")[1] == "addeps":
            tenNormalize = tenNormalize + 0.0000001

        elif strMode.split("-")[1] == "zeroeps":
            tenNormalize[tenNormalize == 0.0] = 1.0

        elif strMode.split("-")[1] == "clipeps":
            tenNormalize = tenNormalize.clip(0.0000001, None)

        if return_norm:
            return tenOut[:, :-1, :, :], tenNormalize

        tenOut = tenOut[:, :-1, :, :] / tenNormalize

    if torch.isnan(tenOut).any():
        print("NaN values detected during training in tenOut_2. Exiting.")
        assert False

    return tenOut
