/// I16x16 intra prediction (H.264 spec 8.3.3).
/// `mode`: 0=vertical, 1=horizontal, 2=DC, 3=plane.
/// For pixels at the top-left of the frame with no neighbors, only DC mode
/// with value 128 is valid.
pub fn predict_intra_16x16(
    mode: u8,
    above: Option<&[u8]>,  // 16 pixels from row above
    left: Option<&[u8]>,   // 16 pixels from column to left
    above_left: Option<u8>,
    output: &mut [u8; 256],
) {
    match mode {
        0 => {
            // Vertical: copy above row to all rows
            let above = above.unwrap_or(&[128; 16]);
            for row in 0..16 {
                output[row * 16..row * 16 + 16].copy_from_slice(&above[..16]);
            }
        }
        1 => {
            // Horizontal: copy left column to all columns
            let left = left.unwrap_or(&[128; 16]);
            for row in 0..16 {
                for col in 0..16 {
                    output[row * 16 + col] = left[row];
                }
            }
        }
        2 => {
            // DC: average of available above and left samples
            let dc = match (above, left) {
                (Some(a), Some(l)) => {
                    let sum: u32 = a[..16].iter().map(|&x| x as u32).sum::<u32>()
                        + l[..16].iter().map(|&x| x as u32).sum::<u32>();
                    ((sum + 16) >> 5) as u8
                }
                (Some(a), None) => {
                    let sum: u32 = a[..16].iter().map(|&x| x as u32).sum();
                    ((sum + 8) >> 4) as u8
                }
                (None, Some(l)) => {
                    let sum: u32 = l[..16].iter().map(|&x| x as u32).sum();
                    ((sum + 8) >> 4) as u8
                }
                (None, None) => 128,
            };
            output.fill(dc);
        }
        3 => {
            // Plane prediction
            let above = above.unwrap_or(&[128; 16]);
            let left = left.unwrap_or(&[128; 16]);
            let _p = above_left.unwrap_or(128);

            let mut h: i32 = 0;
            let mut v: i32 = 0;
            for i in 0..8 {
                // p[6-i, -1] and p[-1, 6-i]: when i==7, index -1 = above_left pixel
                let above_neg = if i < 7 { above[6 - i] as i32 } else { _p as i32 };
                let left_neg = if i < 7 { left[6 - i] as i32 } else { _p as i32 };
                h += (i as i32 + 1) * (above[8 + i] as i32 - above_neg);
                v += (i as i32 + 1) * (left[8 + i] as i32 - left_neg);
            }
            let a_val = 16 * (above[15] as i32 + left[15] as i32);
            let b_val = (5 * h + 32) >> 6;
            let c_val = (5 * v + 32) >> 6;

            for y in 0..16 {
                for x in 0..16 {
                    let val =
                        (a_val + b_val * (x as i32 - 7) + c_val * (y as i32 - 7) + 16) >> 5;
                    output[y * 16 + x] = val.clamp(0, 255) as u8;
                }
            }
        }
        _ => {
            output.fill(128);
        }
    }
}

/// I4x4 intra prediction (H.264 spec 8.3.1.2).
/// `mode`: 0-8 prediction modes.
/// `above`: 8 pixels above (4 directly above + 4 above-right), None if unavailable.
/// `left`: 4 pixels to the left, None if unavailable.
/// `above_left`: pixel diagonally above-left, None if unavailable.
/// `output`: 4x4 block in raster order (16 bytes).
pub fn predict_intra_4x4(
    mode: u8,
    above: Option<&[u8]>,
    left: Option<&[u8]>,
    above_left: Option<u8>,
    output: &mut [u8; 16],
) {
    match mode {
        0 => {
            // Vertical
            let a = above.unwrap_or(&[128; 8]);
            for row in 0..4 {
                output[row * 4..row * 4 + 4].copy_from_slice(&a[..4]);
            }
        }
        1 => {
            // Horizontal
            let l = left.unwrap_or(&[128; 4]);
            for row in 0..4 {
                for col in 0..4 {
                    output[row * 4 + col] = l[row];
                }
            }
        }
        2 => {
            // DC
            let dc = match (above, left) {
                (Some(a), Some(l)) => {
                    let sum: u32 = a[..4].iter().map(|&x| x as u32).sum::<u32>()
                        + l[..4].iter().map(|&x| x as u32).sum::<u32>();
                    ((sum + 4) >> 3) as u8
                }
                (Some(a), None) => {
                    let sum: u32 = a[..4].iter().map(|&x| x as u32).sum();
                    ((sum + 2) >> 2) as u8
                }
                (None, Some(l)) => {
                    let sum: u32 = l[..4].iter().map(|&x| x as u32).sum();
                    ((sum + 2) >> 2) as u8
                }
                (None, None) => 128,
            };
            output.fill(dc);
        }
        3 => {
            // Diagonal Down-Left
            let a = above.unwrap_or(&[128; 8]);
            for y in 0..4 {
                for x in 0..4 {
                    if x == 3 && y == 3 {
                        output[y * 4 + x] =
                            ((a[6] as u16 + 3 * a[7] as u16 + 2) >> 2) as u8;
                    } else {
                        let i = x + y;
                        output[y * 4 + x] = ((a[i] as u16 + 2 * a[i + 1] as u16
                            + a[i + 2] as u16 + 2)
                            >> 2) as u8;
                    }
                }
            }
        }
        4 => {
            // Diagonal Down-Right (spec 8.3.1.2.5)
            // Build reference pixel array: [left[3], left[2], left[1], left[0],
            //                               above_left, above[0..3]]
            // pred[x,y] = (ref[3-y+x] + 2*ref[4-y+x] + ref[5-y+x] + 2) >> 2
            let a = above.unwrap_or(&[128; 8]);
            let l = left.unwrap_or(&[128; 4]);
            let p = above_left.unwrap_or(128);
            let r = [
                l[3] as u16, l[2] as u16, l[1] as u16, l[0] as u16,
                p as u16,
                a[0] as u16, a[1] as u16, a[2] as u16, a[3] as u16,
            ];
            for y in 0..4usize {
                for x in 0..4usize {
                    let i = 3 + x - y;
                    output[y * 4 + x] = ((r[i] + 2 * r[i + 1] + r[i + 2] + 2) >> 2) as u8;
                }
            }
        }
        5 => {
            // Vertical-Right (spec 8.3.1.2.6)
            let a = above.unwrap_or(&[128; 8]);
            let l = left.unwrap_or(&[128; 4]);
            let p = above_left.unwrap_or(128);
            for y in 0..4 {
                for x in 0..4 {
                    let zv = 2 * x as i32 - y as i32;
                    output[y * 4 + x] = if zv >= 0 {
                        let i = x as i32 - (y >> 1) as i32;
                        if zv % 2 == 0 {
                            if i == 0 {
                                ((p as u16 + a[0] as u16 + 1) >> 1) as u8
                            } else {
                                ((a[(i - 1) as usize] as u16 + a[i as usize] as u16 + 1) >> 1)
                                    as u8
                            }
                        } else if i <= 0 {
                            ((l[0] as u16 + 2 * p as u16 + a[0] as u16 + 2) >> 2) as u8
                        } else if i == 1 {
                            ((p as u16 + 2 * a[0] as u16 + a[1] as u16 + 2) >> 2) as u8
                        } else {
                            ((a[(i - 2) as usize] as u16 + 2 * a[(i - 1) as usize] as u16
                                + a[i as usize] as u16
                                + 2)
                                >> 2) as u8
                        }
                    } else if zv == -1 {
                        ((a[0] as u16 + 2 * p as u16 + l[0] as u16 + 2) >> 2) as u8
                    } else {
                        let i = y - 2 * x - 1;
                        let i2 = if i + 1 < 4 { l[i + 1] } else { l[3] };
                        if i == 0 {
                            ((p as u16 + 2 * l[0] as u16 + i2 as u16 + 2) >> 2) as u8
                        } else {
                            ((l[i - 1] as u16 + 2 * l[i] as u16 + i2 as u16 + 2) >> 2) as u8
                        }
                    };
                }
            }
        }
        6 => {
            // Horizontal-Down (spec 8.3.1.2.7)
            let a = above.unwrap_or(&[128; 8]);
            let l = left.unwrap_or(&[128; 4]);
            let p = above_left.unwrap_or(128);
            for y in 0..4 {
                for x in 0..4 {
                    let zh = 2 * y as i32 - x as i32;
                    output[y * 4 + x] = if zh >= 0 {
                        let i = y as i32 - (x >> 1) as i32;
                        if zh % 2 == 0 {
                            if i == 0 {
                                ((p as u16 + l[0] as u16 + 1) >> 1) as u8
                            } else {
                                ((l[(i - 1) as usize] as u16 + l[i as usize] as u16 + 1) >> 1)
                                    as u8
                            }
                        } else if i <= 0 {
                            ((a[0] as u16 + 2 * p as u16 + l[0] as u16 + 2) >> 2) as u8
                        } else if i == 1 {
                            ((p as u16 + 2 * l[0] as u16 + l[1] as u16 + 2) >> 2) as u8
                        } else {
                            ((l[(i - 2) as usize] as u16 + 2 * l[(i - 1) as usize] as u16
                                + l[i as usize] as u16
                                + 2)
                                >> 2) as u8
                        }
                    } else if zh == -1 {
                        ((l[0] as u16 + 2 * p as u16 + a[0] as u16 + 2) >> 2) as u8
                    } else {
                        let i = x - 2 * y - 1;
                        let i2 = if i + 1 < 4 { a[i + 1] } else { a[3] };
                        if i == 0 {
                            ((p as u16 + 2 * a[0] as u16 + i2 as u16 + 2) >> 2) as u8
                        } else {
                            ((a[i - 1] as u16 + 2 * a[i] as u16 + i2 as u16 + 2) >> 2) as u8
                        }
                    };
                }
            }
        }
        7 => {
            // Vertical-Left (spec 8.3.1.2.8)
            let a = above.unwrap_or(&[128; 8]);
            for y in 0..4 {
                for x in 0..4 {
                    let i = x + (y >> 1);
                    output[y * 4 + x] = if y % 2 == 0 {
                        ((a[i] as u16 + a[i + 1] as u16 + 1) >> 1) as u8
                    } else {
                        ((a[i] as u16 + 2 * a[i + 1] as u16 + a[i + 2] as u16 + 2) >> 2) as u8
                    };
                }
            }
        }
        8 => {
            // Horizontal-Up (spec 8.3.1.2.9)
            let l = left.unwrap_or(&[128; 4]);
            for y in 0..4 {
                for x in 0..4 {
                    let zh = x + 2 * y;
                    output[y * 4 + x] = if zh < 5 {
                        let i = y + (x >> 1);
                        if zh % 2 == 0 {
                            ((l[i] as u16 + l[i + 1] as u16 + 1) >> 1) as u8
                        } else {
                            ((l[i] as u16 + 2 * l[i + 1] as u16 + l[i + 2] as u16 + 2) >> 2)
                                as u8
                        }
                    } else if zh == 5 {
                        ((l[2] as u16 + 3 * l[3] as u16 + 2) >> 2) as u8
                    } else {
                        l[3]
                    };
                }
            }
        }
        _ => {
            output.fill(128);
        }
    }
}

/// Chroma 8x8 intra prediction (H.264 spec 8.3.4).
/// `mode`: 0=DC, 1=horizontal, 2=vertical, 3=plane.
pub fn predict_chroma_8x8(
    mode: u8,
    above: Option<&[u8]>,  // 8 pixels from row above
    left: Option<&[u8]>,   // 8 pixels from column to left
    above_left: Option<u8>,
    output: &mut [u8; 64],
) {
    match mode {
        0 => {
            // DC: for 8x8 chroma, prediction is done per 4x4 sub-block
            // with available neighbors
            let dc = match (above, left) {
                (Some(a), Some(l)) => {
                    let sum: u32 = a[..8].iter().map(|&x| x as u32).sum::<u32>()
                        + l[..8].iter().map(|&x| x as u32).sum::<u32>();
                    ((sum + 8) >> 4) as u8
                }
                (Some(a), None) => {
                    let sum: u32 = a[..8].iter().map(|&x| x as u32).sum();
                    ((sum + 4) >> 3) as u8
                }
                (None, Some(l)) => {
                    let sum: u32 = l[..8].iter().map(|&x| x as u32).sum();
                    ((sum + 4) >> 3) as u8
                }
                (None, None) => 128,
            };
            output.fill(dc);
        }
        1 => {
            // Horizontal
            let left = left.unwrap_or(&[128; 16]);
            for row in 0..8 {
                for col in 0..8 {
                    output[row * 8 + col] = left[row];
                }
            }
        }
        2 => {
            // Vertical
            let above = above.unwrap_or(&[128; 16]);
            for row in 0..8 {
                output[row * 8..row * 8 + 8].copy_from_slice(&above[..8]);
            }
        }
        3 => {
            // Plane
            let above = above.unwrap_or(&[128; 16]);
            let left = left.unwrap_or(&[128; 16]);
            let _p = above_left.unwrap_or(128);

            let mut h: i32 = 0;
            let mut v: i32 = 0;
            for i in 0..4 {
                // p[2-i, -1] and p[-1, 2-i]: when i==3, index -1 = above_left pixel
                let above_neg = if i < 3 { above[2 - i] as i32 } else { _p as i32 };
                let left_neg = if i < 3 { left[2 - i] as i32 } else { _p as i32 };
                h += (i as i32 + 1) * (above[4 + i] as i32 - above_neg);
                v += (i as i32 + 1) * (left[4 + i] as i32 - left_neg);
            }
            let a_val = 16 * (above[7] as i32 + left[7] as i32);
            let b_val = (17 * h + 16) >> 5;
            let c_val = (17 * v + 16) >> 5;

            for y in 0..8 {
                for x in 0..8 {
                    let val =
                        (a_val + b_val * (x as i32 - 3) + c_val * (y as i32 - 3) + 16) >> 5;
                    output[y * 8 + x] = val.clamp(0, 255) as u8;
                }
            }
        }
        _ => {
            output.fill(128);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference implementation: compute I4x4 prediction using H.264 spec formulas
    /// with explicit p[x,-1]/p[-1,y]/p[-1,-1] indexing.
    fn spec_predict_4x4(
        mode: u8,
        above: &[u8; 8],
        left: &[u8; 4],
        above_left: u8,
    ) -> [u8; 16] {
        let pa = |x: i32| -> i32 {
            if x == -1 { above_left as i32 } else { above[x as usize] as i32 }
        };
        let pl = |y: i32| -> i32 {
            if y == -1 { above_left as i32 } else { left[y as usize] as i32 }
        };

        let mut out = [0u8; 16];
        for y in 0..4i32 {
            for x in 0..4i32 {
                let v: i32 = match mode {
                    0 => pa(x),
                    1 => pl(y),
                    3 => {
                        if x == 3 && y == 3 {
                            (pa(6) + 3 * pa(7) + 2) >> 2
                        } else {
                            (pa(x + y) + 2 * pa(x + y + 1) + pa(x + y + 2) + 2) >> 2
                        }
                    }
                    4 => {
                        if x > y {
                            (pa(x - y - 2) + 2 * pa(x - y - 1) + pa(x - y) + 2) >> 2
                        } else if x < y {
                            (pl(y - x - 2) + 2 * pl(y - x - 1) + pl(y - x) + 2) >> 2
                        } else {
                            (pa(0) + 2 * pa(-1) + pl(0) + 2) >> 2
                        }
                    }
                    5 => {
                        let zvr = 2 * x - y;
                        if zvr >= 0 && zvr % 2 == 0 {
                            let i = x - (y >> 1);
                            (pa(i - 1) + pa(i) + 1) >> 1
                        } else if zvr >= 0 {
                            let i = x - (y >> 1);
                            (pa(i - 2) + 2 * pa(i - 1) + pa(i) + 2) >> 2
                        } else if zvr == -1 {
                            (pa(0) + 2 * pa(-1) + pl(0) + 2) >> 2
                        } else {
                            (pl(y - 2 * x - 2) + 2 * pl(y - 2 * x - 1) + pl(y - 2 * x) + 2) >> 2
                        }
                    }
                    6 => {
                        let zhd = 2 * y - x;
                        if zhd >= 0 && zhd % 2 == 0 {
                            let i = y - (x >> 1);
                            (pl(i - 1) + pl(i) + 1) >> 1
                        } else if zhd >= 0 {
                            let i = y - (x >> 1);
                            (pl(i - 2) + 2 * pl(i - 1) + pl(i) + 2) >> 2
                        } else if zhd == -1 {
                            (pl(0) + 2 * pl(-1) + pa(0) + 2) >> 2
                        } else {
                            (pa(x - 2 * y - 2) + 2 * pa(x - 2 * y - 1) + pa(x - 2 * y) + 2) >> 2
                        }
                    }
                    7 => {
                        let i = x + (y >> 1);
                        if y % 2 == 0 {
                            (pa(i) + pa(i + 1) + 1) >> 1
                        } else {
                            (pa(i) + 2 * pa(i + 1) + pa(i + 2) + 2) >> 2
                        }
                    }
                    8 => {
                        let zhu = x + 2 * y;
                        if zhu < 5 {
                            let i = y + (x >> 1);
                            if zhu % 2 == 0 {
                                (pl(i) + pl(i + 1) + 1) >> 1
                            } else {
                                (pl(i) + 2 * pl(i + 1) + pl(i + 2) + 2) >> 2
                            }
                        } else if zhu == 5 {
                            (pl(2) + 3 * pl(3) + 2) >> 2
                        } else {
                            pl(3)
                        }
                    }
                    _ => 128,
                };
                out[(y * 4 + x) as usize] = v.clamp(0, 255) as u8;
            }
        }
        out
    }

    #[test]
    fn test_all_directional_modes_against_spec() {
        let above: [u8; 8] = [10, 30, 50, 70, 90, 110, 130, 150];
        let left: [u8; 4] = [20, 60, 100, 140];
        let above_left: u8 = 40;

        for mode in [3u8, 4, 5, 6, 7, 8] {
            let expected = spec_predict_4x4(mode, &above, &left, above_left);
            let mut actual = [0u8; 16];
            predict_intra_4x4(
                mode,
                Some(&above[..]),
                Some(&left[..]),
                Some(above_left),
                &mut actual,
            );
            assert_eq!(
                actual, expected,
                "Mode {} mismatch.\n  actual:   {:?}\n  expected: {:?}",
                mode, actual, expected
            );
        }
    }

    #[test]
    fn test_directional_modes_uniform_input() {
        let above = [128u8; 8];
        let left = [128u8; 4];
        for mode in 0..9u8 {
            let mut output = [0u8; 16];
            predict_intra_4x4(mode, Some(&above[..]), Some(&left[..]), Some(128), &mut output);
            assert!(
                output.iter().all(|&v| v == 128),
                "Mode {} should give all 128 for uniform input, got {:?}",
                mode, output
            );
        }
    }
}
