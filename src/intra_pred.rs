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
