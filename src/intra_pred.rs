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
            let above = above.expect("vertical prediction requires above pixels");
            for row in 0..16 {
                output[row * 16..row * 16 + 16].copy_from_slice(&above[..16]);
            }
        }
        1 => {
            // Horizontal: copy left column to all columns
            let left = left.expect("horizontal prediction requires left pixels");
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
            let above = above.expect("plane prediction requires above pixels");
            let left = left.expect("plane prediction requires left pixels");
            let _p = above_left.expect("plane prediction requires above-left pixel");

            let mut h: i32 = 0;
            let mut v: i32 = 0;
            for i in 0..8 {
                h += (i as i32 + 1) * (above[8 + i] as i32 - above[6 - i] as i32);
                v += (i as i32 + 1) * (left[8 + i] as i32 - left[6 - i] as i32);
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
            let a = above.expect("vertical requires above");
            for row in 0..4 {
                output[row * 4..row * 4 + 4].copy_from_slice(&a[..4]);
            }
        }
        1 => {
            // Horizontal
            let l = left.expect("horizontal requires left");
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
            let a = above.expect("DDL requires above");
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
            // Diagonal Down-Right
            let a = above.expect("DDR requires above");
            let l = left.expect("DDR requires left");
            let p = above_left.expect("DDR requires above-left");
            for y in 0..4i32 {
                for x in 0..4i32 {
                    output[y as usize * 4 + x as usize] = if x > y {
                        let i = (x - y - 1) as usize;
                        if i == 0 && y == 0 {
                            ((p as u16 + 2 * a[0] as u16 + a[1] as u16 + 2) >> 2) as u8
                        } else if y == 0 {
                            ((a[i - 1] as u16 + 2 * a[i] as u16 + a[i + 1] as u16 + 2) >> 2)
                                as u8
                        } else {
                            let ai = (x - y - 1) as usize;
                            ((a[ai] as u16 + 2 * a[ai + 1] as u16 + a[ai + 2] as u16 + 2) >> 2)
                                as u8
                        }
                    } else if y > x {
                        let i = (y - x - 1) as usize;
                        if i == 0 && x == 0 {
                            ((p as u16 + 2 * l[0] as u16 + l[1] as u16 + 2) >> 2) as u8
                        } else if x == 0 {
                            ((l[i - 1] as u16 + 2 * l[i] as u16 + l[i + 1] as u16 + 2) >> 2)
                                as u8
                        } else {
                            let li = (y - x - 1) as usize;
                            ((l[li] as u16 + 2 * l[li + 1] as u16 + l[li + 2] as u16 + 2) >> 2)
                                as u8
                        }
                    } else {
                        // x == y
                        ((a[0] as u16 + 2 * p as u16 + l[0] as u16 + 2) >> 2) as u8
                    };
                }
            }
        }
        5 => {
            // Vertical-Right (spec 8.3.1.2.6)
            let a = above.expect("VR requires above");
            let l = left.expect("VR requires left");
            let p = above_left.expect("VR requires above-left");
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
            let a = above.expect("HD requires above");
            let l = left.expect("HD requires left");
            let p = above_left.expect("HD requires above-left");
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
            let a = above.expect("VL requires above");
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
            let l = left.expect("HU requires left");
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
            let left = left.expect("horizontal prediction requires left pixels");
            for row in 0..8 {
                for col in 0..8 {
                    output[row * 8 + col] = left[row];
                }
            }
        }
        2 => {
            // Vertical
            let above = above.expect("vertical prediction requires above pixels");
            for row in 0..8 {
                output[row * 8..row * 8 + 8].copy_from_slice(&above[..8]);
            }
        }
        3 => {
            // Plane
            let above = above.expect("plane prediction requires above pixels");
            let left = left.expect("plane prediction requires left pixels");
            let _p = above_left.expect("plane prediction requires above-left pixel");

            let mut h: i32 = 0;
            let mut v: i32 = 0;
            for i in 0..4 {
                h += (i as i32 + 1) * (above[4 + i] as i32 - above[2 - i] as i32);
                v += (i as i32 + 1) * (left[4 + i] as i32 - left[2 - i] as i32);
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
