import torch
import triton
import triton.language as tl

@triton.jit
def _triton_corr_forward_kernel(
    fmap1_ptr, fmap2_ptr, coords_ptr, corr_ptr,
    b_stride_f1, c_stride_f1, h_stride_f1, w_stride_f1,
    b_stride_f2, c_stride_f2, h_stride_f2, w_stride_f2,
    b_stride_co, c_stride_co, h_stride_co, w_stride_co,
    b_stride_out, d_stride_out, h_stride_out, w_stride_out,
    B, C, H1, W1, H2, W2, r, rd,
    BLOCK_SIZE_C: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_hw = tl.program_id(1)

    h1 = pid_hw // W1
    w1 = pid_hw % W1

    # Load coordinates
    coord_x_ptr = coords_ptr + pid_b * b_stride_co + 0 * c_stride_co + h1 * h_stride_co + w1 * w_stride_co
    coord_y_ptr = coords_ptr + pid_b * b_stride_co + 1 * c_stride_co + h1 * h_stride_co + w1 * w_stride_co

    x = tl.load(coord_x_ptr)
    y = tl.load(coord_y_ptr)

    # math.floor in triton
    x_floor = tl.math.floor(x)
    y_floor = tl.math.floor(y)
    x0 = x_floor.to(tl.int32)
    y0 = y_floor.to(tl.int32)

    dx = x - x_floor
    dy = y - y_floor

    # Prepare channel offsets
    c_offsets = tl.arange(0, BLOCK_SIZE_C)
    c_mask = c_offsets < C

    # Load fmap1 feature vector for this pixel
    f1_ptrs = fmap1_ptr + pid_b * b_stride_f1 + h1 * h_stride_f1 + w1 * w_stride_f1 + c_offsets * c_stride_f1
    f1 = tl.load(f1_ptrs, mask=c_mask, other=0.0)

    for iy_out in range(rd):
        for ix_out in range(rd):
            # Compute destination index in cost volume (iy_out + rd * ix_out)
            # The original scatter logic writes differently, but maps perfectly to this gather logic:

            # The 4 points to gather from:
            # nw: h2 = y0 - r + iy_out + 1, w2 = x0 - r + ix_out + 1
            # ne: h2 = y0 - r + iy_out + 1, w2 = x0 - r + ix_out
            # sw: h2 = y0 - r + iy_out,     w2 = x0 - r + ix_out + 1
            # se: h2 = y0 - r + iy_out,     w2 = x0 - r + ix_out

            h2_sw_se = y0 - r + iy_out
            h2_nw_ne = y0 - r + iy_out + 1

            w2_ne_se = x0 - r + ix_out
            w2_nw_sw = x0 - r + ix_out + 1

            val = 0.0

            # nw
            mask_nw = (h2_nw_ne >= 0) & (h2_nw_ne < H2) & (w2_nw_sw >= 0) & (w2_nw_sw < W2)
            if mask_nw:
                f2_ptrs_nw = fmap2_ptr + pid_b * b_stride_f2 + h2_nw_ne * h_stride_f2 + w2_nw_sw * w_stride_f2 + c_offsets * c_stride_f2
                f2_nw = tl.load(f2_ptrs_nw, mask=c_mask, other=0.0)
                s_nw = tl.sum(f1 * f2_nw)
                val += s_nw * dy * dx

            # ne
            mask_ne = (h2_nw_ne >= 0) & (h2_nw_ne < H2) & (w2_ne_se >= 0) & (w2_ne_se < W2)
            if mask_ne:
                f2_ptrs_ne = fmap2_ptr + pid_b * b_stride_f2 + h2_nw_ne * h_stride_f2 + w2_ne_se * w_stride_f2 + c_offsets * c_stride_f2
                f2_ne = tl.load(f2_ptrs_ne, mask=c_mask, other=0.0)
                s_ne = tl.sum(f1 * f2_ne)
                val += s_ne * dy * (1.0 - dx)

            # sw
            mask_sw = (h2_sw_se >= 0) & (h2_sw_se < H2) & (w2_nw_sw >= 0) & (w2_nw_sw < W2)
            if mask_sw:
                f2_ptrs_sw = fmap2_ptr + pid_b * b_stride_f2 + h2_sw_se * h_stride_f2 + w2_nw_sw * w_stride_f2 + c_offsets * c_stride_f2
                f2_sw = tl.load(f2_ptrs_sw, mask=c_mask, other=0.0)
                s_sw = tl.sum(f1 * f2_sw)
                val += s_sw * (1.0 - dy) * dx

            # se
            mask_se = (h2_sw_se >= 0) & (h2_sw_se < H2) & (w2_ne_se >= 0) & (w2_ne_se < W2)
            if mask_se:
                f2_ptrs_se = fmap2_ptr + pid_b * b_stride_f2 + h2_sw_se * h_stride_f2 + w2_ne_se * w_stride_f2 + c_offsets * c_stride_f2
                f2_se = tl.load(f2_ptrs_se, mask=c_mask, other=0.0)
                s_se = tl.sum(f1 * f2_se)
                val += s_se * (1.0 - dy) * (1.0 - dx)

            # Write to output
            out_idx = iy_out + rd * ix_out
            out_ptr = corr_ptr + pid_b * b_stride_out + out_idx * d_stride_out + h1 * h_stride_out + w1 * w_stride_out
            tl.store(out_ptr, val)

def triton_corr_forward(fmap1, fmap2, coords, r):
    # fmap1: [B, C, H1, W1]
    # fmap2: [B, C, H2, W2]
    # coords: [B, 2, H1, W1]

    B, C, H1, W1 = fmap1.shape
    B2, C2, H2, W2 = fmap2.shape

    assert B == B2
    assert C == C2

    rd = 2 * r + 1

    corr = torch.empty((B, rd * rd, H1, W1), device=fmap1.device, dtype=fmap1.dtype)

    grid = (B, H1 * W1)

    # Find next power of 2 for channels block size to ensure we can load it all at once efficiently
    BLOCK_SIZE_C = triton.next_power_of_2(C)

    _triton_corr_forward_kernel[grid](
        fmap1, fmap2, coords, corr,
        fmap1.stride(0), fmap1.stride(1), fmap1.stride(2), fmap1.stride(3),
        fmap2.stride(0), fmap2.stride(1), fmap2.stride(2), fmap2.stride(3),
        coords.stride(0), coords.stride(1), coords.stride(2), coords.stride(3),
        corr.stride(0), corr.stride(1), corr.stride(2), corr.stride(3),
        B, C, H1, W1, H2, W2, r, rd,
        BLOCK_SIZE_C=BLOCK_SIZE_C
    )

    return corr
