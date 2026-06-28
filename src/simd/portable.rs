//! Portable SIMD motion-compensation kernels.

use core::simd::prelude::*;
use core::simd::{cmp::SimdOrd, Simd};

use crate::inter_pred::{clip_u8, fir6, fir6_i32};

const LANES_U8: usize = 16;
const LANES_I16: usize = 8;
const LANES_I32: usize = 4;

#[inline(always)]
fn clip_i16_to_u8(v: Simd<i16, LANES_I16>) -> Simd<u8, LANES_I16> {
    v.simd_clamp(Simd::splat(0), Simd::splat(255)).cast::<u8>()
}

#[inline(always)]
fn clip_i32_to_u8(v: Simd<i32, LANES_I32>) -> Simd<u8, LANES_I32> {
    v.simd_clamp(Simd::splat(0), Simd::splat(255)).cast::<u8>()
}

#[inline(always)]
fn avg_u8x8(a: Simd<u8, LANES_I16>, b: Simd<u8, LANES_I16>) -> Simd<u8, LANES_I16> {
    ((a.cast::<u16>() + b.cast::<u16>() + Simd::splat(1)) >> Simd::splat(1)).cast::<u8>()
}

#[inline(always)]
fn fir6_u8_i16(
    s0: Simd<u8, LANES_I16>,
    s1: Simd<u8, LANES_I16>,
    s2: Simd<u8, LANES_I16>,
    s3: Simd<u8, LANES_I16>,
    s4: Simd<u8, LANES_I16>,
    s5: Simd<u8, LANES_I16>,
) -> Simd<i16, LANES_I16> {
    let s0 = s0.cast::<i16>();
    let s1 = s1.cast::<i16>();
    let s2 = s2.cast::<i16>();
    let s3 = s3.cast::<i16>();
    let s4 = s4.cast::<i16>();
    let s5 = s5.cast::<i16>();
    s0 - Simd::splat(5) * s1 + Simd::splat(20) * s2 + Simd::splat(20) * s3 - Simd::splat(5) * s4
        + s5
}

#[inline(always)]
fn fir6_u8_i32(
    s0: Simd<u8, LANES_I32>,
    s1: Simd<u8, LANES_I32>,
    s2: Simd<u8, LANES_I32>,
    s3: Simd<u8, LANES_I32>,
    s4: Simd<u8, LANES_I32>,
    s5: Simd<u8, LANES_I32>,
) -> Simd<i32, LANES_I32> {
    let s0 = s0.cast::<i32>();
    let s1 = s1.cast::<i32>();
    let s2 = s2.cast::<i32>();
    let s3 = s3.cast::<i32>();
    let s4 = s4.cast::<i32>();
    let s5 = s5.cast::<i32>();
    s0 - Simd::splat(5) * s1 + Simd::splat(20) * s2 + Simd::splat(20) * s3 - Simd::splat(5) * s4
        + s5
}

#[inline(always)]
fn fir6_i32x4(
    s0: Simd<i32, LANES_I32>,
    s1: Simd<i32, LANES_I32>,
    s2: Simd<i32, LANES_I32>,
    s3: Simd<i32, LANES_I32>,
    s4: Simd<i32, LANES_I32>,
    s5: Simd<i32, LANES_I32>,
) -> Simd<i32, LANES_I32> {
    s0 - Simd::splat(5) * s1 + Simd::splat(20) * s2 + Simd::splat(20) * s3 - Simd::splat(5) * s4
        + s5
}

#[inline(always)]
fn half_pel_h_i16(src: &[u8], i: usize) -> Simd<u8, LANES_I16> {
    let val = fir6_u8_i16(
        Simd::from_slice(&src[i..i + LANES_I16]),
        Simd::from_slice(&src[i + 1..i + 1 + LANES_I16]),
        Simd::from_slice(&src[i + 2..i + 2 + LANES_I16]),
        Simd::from_slice(&src[i + 3..i + 3 + LANES_I16]),
        Simd::from_slice(&src[i + 4..i + 4 + LANES_I16]),
        Simd::from_slice(&src[i + 5..i + 5 + LANES_I16]),
    );
    clip_i16_to_u8((val + Simd::splat(16)) >> Simd::splat(5))
}

#[inline(always)]
fn half_pel_v_i16(rows: [&[u8]; 6], i: usize) -> Simd<u8, LANES_I16> {
    let val = fir6_u8_i16(
        Simd::from_slice(&rows[0][i..i + LANES_I16]),
        Simd::from_slice(&rows[1][i..i + LANES_I16]),
        Simd::from_slice(&rows[2][i..i + LANES_I16]),
        Simd::from_slice(&rows[3][i..i + LANES_I16]),
        Simd::from_slice(&rows[4][i..i + LANES_I16]),
        Simd::from_slice(&rows[5][i..i + LANES_I16]),
    );
    clip_i16_to_u8((val + Simd::splat(16)) >> Simd::splat(5))
}

#[inline(always)]
fn half_pel_hv_i32(rows: [&[u8]; 6], i: usize) -> Simd<u8, LANES_I32> {
    let h0 = fir6_u8_i32(
        Simd::from_slice(&rows[0][i..i + LANES_I32]),
        Simd::from_slice(&rows[0][i + 1..i + 1 + LANES_I32]),
        Simd::from_slice(&rows[0][i + 2..i + 2 + LANES_I32]),
        Simd::from_slice(&rows[0][i + 3..i + 3 + LANES_I32]),
        Simd::from_slice(&rows[0][i + 4..i + 4 + LANES_I32]),
        Simd::from_slice(&rows[0][i + 5..i + 5 + LANES_I32]),
    );
    let h1 = fir6_u8_i32(
        Simd::from_slice(&rows[1][i..i + LANES_I32]),
        Simd::from_slice(&rows[1][i + 1..i + 1 + LANES_I32]),
        Simd::from_slice(&rows[1][i + 2..i + 2 + LANES_I32]),
        Simd::from_slice(&rows[1][i + 3..i + 3 + LANES_I32]),
        Simd::from_slice(&rows[1][i + 4..i + 4 + LANES_I32]),
        Simd::from_slice(&rows[1][i + 5..i + 5 + LANES_I32]),
    );
    let h2 = fir6_u8_i32(
        Simd::from_slice(&rows[2][i..i + LANES_I32]),
        Simd::from_slice(&rows[2][i + 1..i + 1 + LANES_I32]),
        Simd::from_slice(&rows[2][i + 2..i + 2 + LANES_I32]),
        Simd::from_slice(&rows[2][i + 3..i + 3 + LANES_I32]),
        Simd::from_slice(&rows[2][i + 4..i + 4 + LANES_I32]),
        Simd::from_slice(&rows[2][i + 5..i + 5 + LANES_I32]),
    );
    let h3 = fir6_u8_i32(
        Simd::from_slice(&rows[3][i..i + LANES_I32]),
        Simd::from_slice(&rows[3][i + 1..i + 1 + LANES_I32]),
        Simd::from_slice(&rows[3][i + 2..i + 2 + LANES_I32]),
        Simd::from_slice(&rows[3][i + 3..i + 3 + LANES_I32]),
        Simd::from_slice(&rows[3][i + 4..i + 4 + LANES_I32]),
        Simd::from_slice(&rows[3][i + 5..i + 5 + LANES_I32]),
    );
    let h4 = fir6_u8_i32(
        Simd::from_slice(&rows[4][i..i + LANES_I32]),
        Simd::from_slice(&rows[4][i + 1..i + 1 + LANES_I32]),
        Simd::from_slice(&rows[4][i + 2..i + 2 + LANES_I32]),
        Simd::from_slice(&rows[4][i + 3..i + 3 + LANES_I32]),
        Simd::from_slice(&rows[4][i + 4..i + 4 + LANES_I32]),
        Simd::from_slice(&rows[4][i + 5..i + 5 + LANES_I32]),
    );
    let h5 = fir6_u8_i32(
        Simd::from_slice(&rows[5][i..i + LANES_I32]),
        Simd::from_slice(&rows[5][i + 1..i + 1 + LANES_I32]),
        Simd::from_slice(&rows[5][i + 2..i + 2 + LANES_I32]),
        Simd::from_slice(&rows[5][i + 3..i + 3 + LANES_I32]),
        Simd::from_slice(&rows[5][i + 4..i + 4 + LANES_I32]),
        Simd::from_slice(&rows[5][i + 5..i + 5 + LANES_I32]),
    );
    clip_i32_to_u8((fir6_i32x4(h0, h1, h2, h3, h4, h5) + Simd::splat(512)) >> Simd::splat(10))
}

/// Row-based horizontal half-pel filter for in-bounds blocks.
/// Reads `w` output pixels from row at `src` (which must have `w + 5` accessible bytes).
///
/// # Pipeline / lanes
///
/// Processes 8 outputs at a time with six shifted `u8x8` loads, widens to `i16x8`,
/// applies the 6-tap FIR, rounds, shifts, clamps, and stores a scalar tail.
#[inline(always)]
pub fn row_half_pel_h(src: &[u8], out: &mut [u8], w: usize) {
    let mut i = 0;
    while i + LANES_I16 <= w {
        half_pel_h_i16(src, i).copy_to_slice(&mut out[i..i + LANES_I16]);
        i += LANES_I16;
    }
    while i < w {
        out[i] = clip_u8((fir6(src, i) + 16) >> 5);
        i += 1;
    }
}

/// Row-based vertical half-pel filter for in-bounds blocks.
/// `rows` contains 6 row slices (y-2..y+3), each at least `w` bytes.
///
/// # Pipeline / lanes
///
/// Processes 8 columns at a time with one `u8x8` load from each source row, widens
/// to `i16x8`, applies the vertical FIR, rounds, clamps, and stores a scalar tail.
#[inline(always)]
pub fn row_half_pel_v(rows: [&[u8]; 6], out: &mut [u8], w: usize) {
    let mut i = 0;
    while i + LANES_I16 <= w {
        half_pel_v_i16(rows, i).copy_to_slice(&mut out[i..i + LANES_I16]);
        i += LANES_I16;
    }
    while i < w {
        let val = rows[0][i] as i32 - 5 * rows[1][i] as i32
            + 20 * rows[2][i] as i32
            + 20 * rows[3][i] as i32
            - 5 * rows[4][i] as i32
            + rows[5][i] as i32;
        out[i] = clip_u8((val + 16) >> 5);
        i += 1;
    }
}

/// Row-based diagonal half-pel (hv) for in-bounds blocks.
/// `rows` contains 6 row slices (y-2..y+3), each with `w + 5` accessible bytes.
///
/// # Pipeline / lanes
///
/// Processes 4 outputs at a time. Horizontal 6-tap intermediates are kept in
/// `i32x4` before the vertical 6-tap because the second pass can exceed `i16`.
#[inline(always)]
pub fn row_half_pel_hv(rows: [&[u8]; 6], out: &mut [u8], w: usize) {
    let mut i = 0;
    while i + LANES_I32 <= w {
        half_pel_hv_i32(rows, i).copy_to_slice(&mut out[i..i + LANES_I32]);
        i += LANES_I32;
    }
    while i < w {
        let h0 = fir6(rows[0], i);
        let h1 = fir6(rows[1], i);
        let h2 = fir6(rows[2], i);
        let h3 = fir6(rows[3], i);
        let h4 = fir6(rows[4], i);
        let h5 = fir6(rows[5], i);
        out[i] = clip_u8((fir6_i32(h0, h1, h2, h3, h4, h5) + 512) >> 10);
        i += 1;
    }
}

/// Average an integer row with a vertical half-pel row.
///
/// # Pipeline / lanes
///
/// Computes vertical half-pel in `i16x8`, then performs the H.264 rounding
/// average in widened `u16x8` lanes before a scalar tail.
pub fn row_avg_int_v(int_row: &[u8], rows: [&[u8]; 6], out: &mut [u8], w: usize) {
    let mut i = 0;
    while i + LANES_I16 <= w {
        let int_v = Simd::<u8, LANES_I16>::from_slice(&int_row[i..i + LANES_I16]);
        avg_u8x8(int_v, half_pel_v_i16(rows, i)).copy_to_slice(&mut out[i..i + LANES_I16]);
        i += LANES_I16;
    }
    while i < w {
        let val = rows[0][i] as i32 - 5 * rows[1][i] as i32
            + 20 * rows[2][i] as i32
            + 20 * rows[3][i] as i32
            - 5 * rows[4][i] as i32
            + rows[5][i] as i32;
        let hp = clip_u8((val + 16) >> 5);
        out[i] = ((int_row[i] as u16 + hp as u16 + 1) >> 1) as u8;
        i += 1;
    }
}

/// Average horizontal and vertical half-pel rows.
///
/// # Pipeline / lanes
///
/// Computes both half-pel rows in `i16x8` lanes, clips to `u8x8`, then uses the
/// widened rounding average required by the scalar reference.
pub fn row_avg_h_v(src_h: &[u8], rows_v: [&[u8]; 6], out: &mut [u8], w: usize) {
    let mut i = 0;
    while i + LANES_I16 <= w {
        avg_u8x8(half_pel_h_i16(src_h, i), half_pel_v_i16(rows_v, i))
            .copy_to_slice(&mut out[i..i + LANES_I16]);
        i += LANES_I16;
    }
    while i < w {
        let h_val = clip_u8((fir6(src_h, i) + 16) >> 5);
        let v_val = clip_u8(
            (rows_v[0][i] as i32 - 5 * rows_v[1][i] as i32
                + 20 * rows_v[2][i] as i32
                + 20 * rows_v[3][i] as i32
                - 5 * rows_v[4][i] as i32
                + rows_v[5][i] as i32
                + 16)
                >> 5,
        );
        out[i] = ((h_val as u16 + v_val as u16 + 1) >> 1) as u8;
        i += 1;
    }
}

/// Chroma bilinear interpolation for an in-bounds block.
///
/// # Pipeline / lanes
///
/// Vectorizes the inner row loop in `i32x4` lanes, splatting the bilinear
/// coefficients and preserving the scalar `(sum + 32) >> 6` rounding.
#[allow(clippy::too_many_arguments)]
pub fn chroma_bilinear_block(
    ref_plane: &[u8],
    top_off: usize,
    ref_width: usize,
    block_w: usize,
    block_h: usize,
    output: &mut [u8],
    c00: u8,
    c01: u8,
    c10: u8,
    c11: u8,
) {
    let c00 = Simd::<i32, LANES_I32>::splat(c00 as i32);
    let c01 = Simd::<i32, LANES_I32>::splat(c01 as i32);
    let c10 = Simd::<i32, LANES_I32>::splat(c10 as i32);
    let c11 = Simd::<i32, LANES_I32>::splat(c11 as i32);

    for row in 0..block_h {
        let row_top = top_off + row * ref_width;
        let row_bot = row_top + ref_width;
        let out_off = row * block_w;
        let mut i = 0;
        while i + LANES_I32 <= block_w {
            let p00 =
                Simd::<u8, LANES_I32>::from_slice(&ref_plane[row_top + i..row_top + i + LANES_I32])
                    .cast::<i32>();
            let p01 = Simd::<u8, LANES_I32>::from_slice(
                &ref_plane[row_top + i + 1..row_top + i + 1 + LANES_I32],
            )
            .cast::<i32>();
            let p10 =
                Simd::<u8, LANES_I32>::from_slice(&ref_plane[row_bot + i..row_bot + i + LANES_I32])
                    .cast::<i32>();
            let p11 = Simd::<u8, LANES_I32>::from_slice(
                &ref_plane[row_bot + i + 1..row_bot + i + 1 + LANES_I32],
            )
            .cast::<i32>();
            let val =
                (c00 * p00 + c01 * p01 + c10 * p10 + c11 * p11 + Simd::splat(32)) >> Simd::splat(6);
            val.cast::<u8>()
                .copy_to_slice(&mut output[out_off + i..out_off + i + LANES_I32]);
            i += LANES_I32;
        }
        while i < block_w {
            let val = c00[0] * ref_plane[row_top + i] as i32
                + c01[0] * ref_plane[row_top + i + 1] as i32
                + c10[0] * ref_plane[row_bot + i] as i32
                + c11[0] * ref_plane[row_bot + i + 1] as i32;
            output[out_off + i] = ((val + 32) >> 6) as u8;
            i += 1;
        }
    }
}

/// Bi-prediction averaging (spec 8.4.2.3.2).
/// `output[i] = (pred_l0[i] + pred_l1[i] + 1) >> 1` for each pixel.
///
/// # Pipeline / lanes
///
/// Processes 16 pixels per iteration by widening `u8x16` inputs to `u16x16`,
/// adding the scalar rounding bias, shifting, narrowing, and tailing scalar.
pub fn bi_pred_avg(pred_l0: &[u8], pred_l1: &[u8], output: &mut [u8]) {
    let len = output.len().min(pred_l0.len()).min(pred_l1.len());
    let mut i = 0;
    while i + LANES_U8 <= len {
        let a = Simd::<u8, LANES_U8>::from_slice(&pred_l0[i..i + LANES_U8]).cast::<u16>();
        let b = Simd::<u8, LANES_U8>::from_slice(&pred_l1[i..i + LANES_U8]).cast::<u16>();
        ((a + b + Simd::splat(1)) >> Simd::splat(1))
            .cast::<u8>()
            .copy_to_slice(&mut output[i..i + LANES_U8]);
        i += LANES_U8;
    }
    while i < len {
        output[i] = ((pred_l0[i] as u16 + pred_l1[i] as u16 + 1) >> 1) as u8;
        i += 1;
    }
}

/// Apply explicit weighted prediction to uni-directional MC output (spec 8.4.2.3.1).
/// `output[i] = clip((pred[i] * weight + (1 << (log2_denom - 1))) >> log2_denom + offset)`
/// When log2_denom == 0, the rounding term is 0.
///
/// # Pipeline / lanes
///
/// Uses `i32x4` lanes for multiply/add/shift/clamp and mirrors the scalar two-branch
/// rounding and `wrapping_add(offset)` order exactly.
pub fn weighted_uni(pred: &mut [u8], log2_denom: u32, weight: i32, offset: i32) {
    let weight_v = Simd::<i32, LANES_I32>::splat(weight);
    let offset_v = Simd::<i32, LANES_I32>::splat(offset);
    let mut i = 0;
    if log2_denom == 0 {
        while i + LANES_I32 <= pred.len() {
            let p = Simd::<u8, LANES_I32>::from_slice(&pred[i..i + LANES_I32]).cast::<i32>();
            clip_i32_to_u8(p * weight_v + offset_v).copy_to_slice(&mut pred[i..i + LANES_I32]);
            i += LANES_I32;
        }
        while i < pred.len() {
            pred[i] = ((pred[i] as i32 * weight + offset).clamp(0, 255)) as u8;
            i += 1;
        }
    } else {
        let round_v = Simd::<i32, LANES_I32>::splat(1i32 << (log2_denom - 1));
        let shift_v = Simd::<i32, LANES_I32>::splat(log2_denom as i32);
        while i + LANES_I32 <= pred.len() {
            let p = Simd::<u8, LANES_I32>::from_slice(&pred[i..i + LANES_I32]).cast::<i32>();
            clip_i32_to_u8(((p * weight_v + round_v) >> shift_v) + offset_v)
                .copy_to_slice(&mut pred[i..i + LANES_I32]);
            i += LANES_I32;
        }
        while i < pred.len() {
            pred[i] = ((pred[i] as i32 * weight + (1i32 << (log2_denom - 1))) >> log2_denom)
                .wrapping_add(offset)
                .clamp(0, 255) as u8;
            i += 1;
        }
    }
}

/// Apply explicit weighted bi-prediction (spec 8.4.2.3.2).
/// Formula: `clip((p0*w0 + p1*w1 + round) >> (denom+1) + (o0+o1+1)>>1)`
///
/// # Pipeline / lanes
///
/// Uses `i32x4` lanes for both weighted inputs, applies the scalar shift, then
/// performs the required wrapping offset addition before clamping.
#[allow(clippy::too_many_arguments)]
pub fn weighted_bi(
    pred_l0: &[u8],
    pred_l1: &[u8],
    output: &mut [u8],
    log2_denom: u32,
    w0: i32,
    o0: i32,
    w1: i32,
    o1: i32,
) {
    let len = output.len().min(pred_l0.len()).min(pred_l1.len());
    let round_v = Simd::<i32, LANES_I32>::splat(1i32 << log2_denom);
    let offset_v = Simd::<i32, LANES_I32>::splat((o0 + o1 + 1) >> 1);
    let shift_v = Simd::<i32, LANES_I32>::splat((log2_denom + 1) as i32);
    let w0_v = Simd::<i32, LANES_I32>::splat(w0);
    let w1_v = Simd::<i32, LANES_I32>::splat(w1);
    let mut i = 0;
    while i + LANES_I32 <= len {
        let a = Simd::<u8, LANES_I32>::from_slice(&pred_l0[i..i + LANES_I32]).cast::<i32>();
        let b = Simd::<u8, LANES_I32>::from_slice(&pred_l1[i..i + LANES_I32]).cast::<i32>();
        clip_i32_to_u8(((a * w0_v + b * w1_v + round_v) >> shift_v) + offset_v)
            .copy_to_slice(&mut output[i..i + LANES_I32]);
        i += LANES_I32;
    }
    while i < len {
        output[i] = ((pred_l0[i] as i32 * w0 + pred_l1[i] as i32 * w1 + (1i32 << log2_denom))
            >> (log2_denom + 1))
            .wrapping_add((o0 + o1 + 1) >> 1)
            .clamp(0, 255) as u8;
        i += 1;
    }
}

/// Apply implicit weighted bi-prediction for B-slices (spec 8.4.2.3.2).
/// Uses POC-distance-derived weights with fixed log2_denom=5.
///
/// # Pipeline / lanes
///
/// Uses `i32x4` lanes for the two weighted inputs and fixed `(sum + 32) >> 6`
/// rounding before clamping.
pub fn weighted_bi_implicit(pred_l0: &[u8], pred_l1: &[u8], output: &mut [u8], w0: i32, w1: i32) {
    let len = output.len().min(pred_l0.len()).min(pred_l1.len());
    let w0_v = Simd::<i32, LANES_I32>::splat(w0);
    let w1_v = Simd::<i32, LANES_I32>::splat(w1);
    let mut i = 0;
    while i + LANES_I32 <= len {
        let a = Simd::<u8, LANES_I32>::from_slice(&pred_l0[i..i + LANES_I32]).cast::<i32>();
        let b = Simd::<u8, LANES_I32>::from_slice(&pred_l1[i..i + LANES_I32]).cast::<i32>();
        clip_i32_to_u8((a * w0_v + b * w1_v + Simd::splat(1i32 << 5)) >> Simd::splat(6))
            .copy_to_slice(&mut output[i..i + LANES_I32]);
        i += LANES_I32;
    }
    while i < len {
        output[i] = ((pred_l0[i] as i32 * w0 + pred_l1[i] as i32 * w1 + (1i32 << 5)) >> 6)
            .clamp(0, 255) as u8;
        i += 1;
    }
}
