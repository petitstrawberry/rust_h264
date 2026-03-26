//! H.264 inter prediction / motion compensation (spec 8.4.2).
//!
//! Generates predicted blocks for P/B slices by interpolating pixels from
//! reference frames at quarter-pel (luma) or eighth-pel (chroma) precision.

use crate::dpb::DecodedPicture;

/// Fetch a luma sample from the reference picture with boundary clipping.
/// Out-of-bounds coordinates are clamped to the picture edge (spec 8.4.2.2.1).
#[inline]
fn ref_luma(pic: &DecodedPicture, x: i32, y: i32) -> i32 {
    let cx = x.clamp(0, pic.width as i32 - 1) as usize;
    let cy = y.clamp(0, pic.height as i32 - 1) as usize;
    pic.y[cy * pic.width as usize + cx] as i32
}

/// Fetch a chroma sample with boundary clipping.
#[inline]
fn ref_chroma(plane: &[u8], width: usize, height: usize, x: i32, y: i32) -> i32 {
    let cx = x.clamp(0, width as i32 - 1) as usize;
    let cy = y.clamp(0, height as i32 - 1) as usize;
    plane[cy * width + cx] as i32
}

#[inline]
fn clip_u8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

#[inline]
fn avg(a: u8, b: u8) -> u8 {
    ((a as u16 + b as u16 + 1) >> 1) as u8
}

/// 6-tap horizontal half-pel filter at integer position (x, y).
/// Returns clipped u8 result.
fn half_pel_h(pic: &DecodedPicture, x: i32, y: i32) -> u8 {
    let s = |dx: i32| ref_luma(pic, x + dx, y);
    clip_u8((s(-2) - 5 * s(-1) + 20 * s(0) + 20 * s(1) - 5 * s(2) + s(3) + 16) >> 5)
}

/// 6-tap vertical half-pel filter at integer position (x, y).
fn half_pel_v(pic: &DecodedPicture, x: i32, y: i32) -> u8 {
    let s = |dy: i32| ref_luma(pic, x, y + dy);
    clip_u8((s(-2) - 5 * s(-1) + 20 * s(0) + 20 * s(1) - 5 * s(2) + s(3) + 16) >> 5)
}

/// Diagonal half-pel: 6-tap horizontal on 6 rows, then 6-tap vertical on
/// the UNCLIPPED intermediates. Final result clipped after >> 10.
fn half_pel_hv(pic: &DecodedPicture, x: i32, y: i32) -> u8 {
    // First pass: horizontal filter on 6 vertically adjacent rows
    let mut h = [0i32; 6];
    for (i, dy) in (-2..=3).enumerate() {
        let s = |dx: i32| ref_luma(pic, x + dx, y + dy);
        h[i] = s(-2) - 5 * s(-1) + 20 * s(0) + 20 * s(1) - 5 * s(2) + s(3);
        // NOT clipped — intermediates stay as i32
    }
    // Second pass: vertical filter on intermediates
    let val = h[0] - 5 * h[1] + 20 * h[2] + 20 * h[3] - 5 * h[4] + h[5];
    clip_u8((val + 512) >> 10)
}

/// Interpolate a single luma sample at quarter-pel position.
/// `x`, `y` are integer-pel coordinates of the reference position.
/// `frac_x`, `frac_y` are the fractional offsets (0..3).
fn luma_interp(
    pic: &DecodedPicture,
    x: i32,
    y: i32,
    frac_x: i32,
    frac_y: i32,
) -> u8 {
    // Spec 8.4.2.2.1: 16 fractional positions (4x4 grid)
    match (frac_x, frac_y) {
        // Integer position
        (0, 0) => ref_luma(pic, x, y) as u8,

        // Half-pel positions
        (2, 0) => half_pel_h(pic, x, y),
        (0, 2) => half_pel_v(pic, x, y),
        (2, 2) => half_pel_hv(pic, x, y),

        // Quarter-pel horizontal (average of integer and half-pel)
        (1, 0) => avg(ref_luma(pic, x, y) as u8, half_pel_h(pic, x, y)),
        (3, 0) => avg(half_pel_h(pic, x, y), ref_luma(pic, x + 1, y) as u8),

        // Quarter-pel vertical
        (0, 1) => avg(ref_luma(pic, x, y) as u8, half_pel_v(pic, x, y)),
        (0, 3) => avg(half_pel_v(pic, x, y), ref_luma(pic, x, y + 1) as u8),

        // Quarter-pel diagonal: average of half-pel neighbors
        // Spec defines these as average of the two nearest half-pel samples
        (2, 1) => avg(half_pel_h(pic, x, y), half_pel_hv(pic, x, y)),
        (2, 3) => avg(half_pel_hv(pic, x, y), half_pel_h(pic, x, y + 1)),
        (1, 2) => avg(half_pel_v(pic, x, y), half_pel_hv(pic, x, y)),
        (3, 2) => avg(half_pel_hv(pic, x, y), half_pel_v(pic, x + 1, y)),

        // Quarter-pel corner positions: average of integer corner and diagonal half-pel
        (1, 1) => avg(ref_luma(pic, x, y) as u8, half_pel_hv(pic, x, y)),
        (3, 1) => avg(ref_luma(pic, x + 1, y) as u8, half_pel_hv(pic, x, y)),
        (1, 3) => avg(ref_luma(pic, x, y + 1) as u8, half_pel_hv(pic, x, y)),
        (3, 3) => avg(ref_luma(pic, x + 1, y + 1) as u8, half_pel_hv(pic, x, y)),

        _ => unreachable!(),
    }
}

/// Perform luma motion compensation for a block.
///
/// `x`, `y`: block top-left in full-pel picture coordinates.
/// `dx`, `dy`: motion vector in quarter-pel units.
/// `block_w`, `block_h`: block dimensions (4, 8, or 16).
/// `output`: predicted pixels, length = block_w * block_h.
#[allow(clippy::too_many_arguments)]
pub fn luma_mc(
    ref_pic: &DecodedPicture,
    x: i32,
    y: i32,
    dx: i32,
    dy: i32,
    block_w: usize,
    block_h: usize,
    output: &mut [u8],
) {
    let frac_x = dx.rem_euclid(4);
    let frac_y = dy.rem_euclid(4);
    // Integer part: arithmetic right shift gives floor division for negative values
    let x_int = x + (dx >> 2);
    let y_int = y + (dy >> 2);

    for row in 0..block_h {
        for col in 0..block_w {
            output[row * block_w + col] = luma_interp(
                ref_pic,
                x_int + col as i32,
                y_int + row as i32,
                frac_x,
                frac_y,
            );
        }
    }
}

/// Perform chroma motion compensation for one plane (U or V).
///
/// Chroma MVs use the same quarter-pel values as luma, but since chroma is
/// half spatial resolution (4:2:0), these become eighth-pel for chroma.
/// Uses bilinear interpolation (spec 8.4.2.2.2).
///
/// `x`, `y`: chroma block top-left in full chroma-pel coordinates.
/// `dx`, `dy`: motion vector in eighth-pel units (= luma quarter-pel MV).
#[allow(clippy::too_many_arguments)]
pub fn chroma_mc(
    ref_plane: &[u8],
    ref_width: usize,
    ref_height: usize,
    x: i32,
    y: i32,
    dx: i32,
    dy: i32,
    block_w: usize,
    block_h: usize,
    output: &mut [u8],
) {
    let frac_x = dx.rem_euclid(8);
    let frac_y = dy.rem_euclid(8);
    let x_int = x + (dx >> 3);
    let y_int = y + (dy >> 3);

    for row in 0..block_h {
        for col in 0..block_w {
            let xf = x_int + col as i32;
            let yf = y_int + row as i32;
            let a = ref_chroma(ref_plane, ref_width, ref_height, xf, yf);
            let b = ref_chroma(ref_plane, ref_width, ref_height, xf + 1, yf);
            let c = ref_chroma(ref_plane, ref_width, ref_height, xf, yf + 1);
            let d = ref_chroma(ref_plane, ref_width, ref_height, xf + 1, yf + 1);

            let val = (8 - frac_x) * (8 - frac_y) * a
                + frac_x * (8 - frac_y) * b
                + (8 - frac_x) * frac_y * c
                + frac_x * frac_y * d;
            output[row * block_w + col] = ((val + 32) >> 6) as u8;
        }
    }
}

/// Bi-prediction averaging (spec 8.4.2.3.2).
/// `output[i] = (pred_l0[i] + pred_l1[i] + 1) >> 1` for each pixel.
pub fn bi_pred_avg(pred_l0: &[u8], pred_l1: &[u8], output: &mut [u8]) {
    for (o, (&a, &b)) in output.iter_mut().zip(pred_l0.iter().zip(pred_l1.iter())) {
        *o = ((a as u16 + b as u16 + 1) >> 1) as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::rc::Rc;

    fn make_ref_pic(width: u32, height: u32, y_data: Vec<u8>) -> Rc<DecodedPicture> {
        let uv_size = (width / 2 * height / 2) as usize;
        Rc::new(DecodedPicture {
            y: y_data,
            u: vec![128; uv_size],
            v: vec![128; uv_size],
            width,
            height,
            frame_num: 0,
            pic_order_cnt: 0,
            mv_l0: vec![],
            ref_idx_l0: vec![],
            mb_width: width / 16,
            is_intra: false,
        })
    }

    #[test]
    fn test_integer_pel_copy() {
        // 8x8 reference with gradient
        let mut y = vec![0u8; 64];
        for r in 0..8 {
            for c in 0..8 {
                y[r * 8 + c] = (r * 16 + c * 4) as u8;
            }
        }
        let pic = make_ref_pic(8, 8, y.clone());

        // Integer-pel MC (dx=0, dy=0) should be a direct copy
        let mut out = vec![0u8; 16]; // 4x4 block at (2,1)
        luma_mc(&pic, 1, 2, 0, 0, 4, 4, &mut out);
        for r in 0..4 {
            for c in 0..4 {
                assert_eq!(out[r * 4 + c], y[(r + 2) * 8 + (c + 1)]);
            }
        }
    }

    #[test]
    fn test_integer_pel_with_mv() {
        // 8x8 uniform reference
        let y = vec![42u8; 64];
        let pic = make_ref_pic(8, 8, y);

        // MV = (4, 8) in quarter-pel = (1, 2) full-pel offset
        let mut out = vec![0u8; 16];
        luma_mc(&pic, 0, 0, 4, 8, 4, 4, &mut out);
        // With uniform reference, all outputs should be 42
        assert!(out.iter().all(|&v| v == 42));
    }

    #[test]
    fn test_half_pel_horizontal() {
        // 16x1 reference: known values for 6-tap filter
        let mut y = vec![128u8; 16];
        // Set a step edge: left half = 0, right half = 255
        for c in 0..8 {
            y[c] = 0;
        }
        for c in 8..16 {
            y[c] = 255;
        }
        let pic = make_ref_pic(16, 1, y);

        // Half-pel horizontal at x=7 (the edge): MV dx=2 (half-pel), dy=0
        let mut out = [0u8; 1];
        luma_mc(&pic, 7, 0, 2, 0, 1, 1, &mut out);
        // 6-tap at x=7: samples at x=5..10 = [0, 0, 0, 255, 255, 255]
        // (0 - 0 + 0 + 20*255 - 5*255 + 255 + 16) >> 5 = (0 + 5100 - 1275 + 255 + 16) >> 5
        // = 4096 >> 5 = 128
        assert_eq!(out[0], 128);
    }

    #[test]
    fn test_half_pel_vertical() {
        // 1x16 reference with step edge at row 8
        let mut y = vec![0u8; 16];
        for r in 8..16 {
            y[r] = 255;
        }
        let pic = make_ref_pic(1, 16, y);

        let mut out = [0u8; 1];
        luma_mc(&pic, 0, 7, 0, 2, 1, 1, &mut out);
        // Same as horizontal but vertical: should also give 128
        assert_eq!(out[0], 128);
    }

    #[test]
    fn test_uniform_ref_all_frac_positions() {
        // Uniform reference: all fractional positions should give the same value
        let y = vec![100u8; 256];
        let pic = make_ref_pic(16, 16, y);

        for frac_x in 0..4 {
            for frac_y in 0..4 {
                let mut out = [0u8; 1];
                luma_mc(&pic, 4, 4, frac_x, frac_y, 1, 1, &mut out);
                assert_eq!(
                    out[0], 100,
                    "frac ({},{}) should give 100 for uniform ref",
                    frac_x, frac_y
                );
            }
        }
    }

    #[test]
    fn test_chroma_mc_integer() {
        let plane = vec![200u8; 64]; // 8x8 chroma
        let mut out = vec![0u8; 16]; // 4x4 block
        chroma_mc(&plane, 8, 8, 0, 0, 0, 0, 4, 4, &mut out);
        assert!(out.iter().all(|&v| v == 200));
    }

    #[test]
    fn test_chroma_mc_half_pel() {
        // 4x1 chroma: [0, 255, 0, 255]
        let plane = vec![0, 255, 0, 255];
        let mut out = [0u8; 1];
        // dx=4 = half-pel (4/8 = 0.5), dy=0
        chroma_mc(&plane, 4, 1, 0, 0, 4, 0, 1, 1, &mut out);
        // Bilinear: (8-4)*8*0 + 4*8*255 + 0 + 0 = 8160. (8160+32)>>6 = 128 (approx)
        assert_eq!(out[0], 128);
    }

    #[test]
    fn test_boundary_clipping() {
        // 4x4 reference, MV pointing outside
        let y: Vec<u8> = (0..16).collect();
        let pic = make_ref_pic(4, 4, y);

        // MC at (0,0) with MV=(-4, -4) in quarter-pel = (-1, -1) full-pel
        // Should clamp to (0,0) and read the top-left corner value
        let mut out = [0u8; 1];
        luma_mc(&pic, 0, 0, -4, -4, 1, 1, &mut out);
        assert_eq!(out[0], 0); // clamped to (0,0)
    }

    #[test]
    fn test_negative_mv_fractional() {
        // Verify negative MV fractional extraction
        let y = vec![128u8; 256];
        let pic = make_ref_pic(16, 16, y);

        // MV = (-1, -1) quarter-pel: frac should be (3, 3), int offset = (-1, -1)
        let mut out = [0u8; 1];
        luma_mc(&pic, 8, 8, -1, -1, 1, 1, &mut out);
        // Uniform ref, so result should be 128 regardless
        assert_eq!(out[0], 128);
    }
}
