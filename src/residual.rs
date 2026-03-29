/// 4x4 zigzag scan order: maps linear index to (row, col) within a 4x4 block.
pub const ZIGZAG_4X4: [(usize, usize); 16] = [
    (0, 0), (0, 1), (1, 0), (2, 0),
    (1, 1), (0, 2), (0, 3), (1, 2),
    (2, 1), (3, 0), (3, 1), (2, 2),
    (1, 3), (2, 3), (3, 2), (3, 3),
];

/// Inverse 4x4 Hadamard transform for I16x16 luma DC coefficients.
/// Input/output: 16 values arranged as 4x4 in raster order.
pub fn inverse_hadamard_4x4(dc: &mut [i32; 16]) {
    let mut tmp = [0i32; 16];

    // Horizontal transform
    for i in 0..4 {
        let a = dc[i * 4];
        let b = dc[i * 4 + 1];
        let c = dc[i * 4 + 2];
        let d = dc[i * 4 + 3];
        tmp[i * 4] = a + b + c + d;
        tmp[i * 4 + 1] = a + b - c - d;
        tmp[i * 4 + 2] = a - b - c + d;
        tmp[i * 4 + 3] = a - b + c - d;
    }

    // Vertical transform
    for j in 0..4 {
        let a = tmp[j];
        let b = tmp[4 + j];
        let c = tmp[8 + j];
        let d = tmp[12 + j];
        dc[j] = a + b + c + d;
        dc[4 + j] = a + b - c - d;
        dc[8 + j] = a - b - c + d;
        dc[12 + j] = a - b + c - d;
    }
}

/// Inverse 2x2 Hadamard transform for chroma DC coefficients.
pub fn inverse_hadamard_2x2(dc: &mut [i32; 4]) {
    let a = dc[0] + dc[1] + dc[2] + dc[3];
    let b = dc[0] - dc[1] + dc[2] - dc[3];
    let c = dc[0] + dc[1] - dc[2] - dc[3];
    let d = dc[0] - dc[1] - dc[2] + dc[3];
    dc[0] = a;
    dc[1] = b;
    dc[2] = c;
    dc[3] = d;
}

/// Inverse 4x4 integer DCT transform (H.264 spec 8.5.12).
/// Operates in-place on 16 coefficients in raster order.
/// Horizontal pass (rows) first, then vertical pass (columns), per spec 8.5.12.1.
pub fn inverse_dct_4x4(block: &mut [i32; 16]) {
    let mut tmp = [0i32; 16];

    // Horizontal pass (rows)
    for i in 0..4 {
        let z0 = block[i * 4];
        let z1 = block[i * 4 + 1];
        let z2 = block[i * 4 + 2];
        let z3 = block[i * 4 + 3];

        let e0 = z0 + z2;
        let e1 = z0 - z2;
        let e2 = (z1 >> 1) - z3;
        let e3 = z1 + (z3 >> 1);

        tmp[i * 4] = e0 + e3;
        tmp[i * 4 + 1] = e1 + e2;
        tmp[i * 4 + 2] = e1 - e2;
        tmp[i * 4 + 3] = e0 - e3;
    }

    // Vertical pass (columns), with rounding: (x + 32) >> 6
    for j in 0..4 {
        let z0 = tmp[j];
        let z1 = tmp[4 + j];
        let z2 = tmp[8 + j];
        let z3 = tmp[12 + j];

        let e0 = z0 + z2;
        let e1 = z0 - z2;
        let e2 = (z1 >> 1) - z3;
        let e3 = z1 + (z3 >> 1);

        block[j] = (e0 + e3 + 32) >> 6;
        block[4 + j] = (e1 + e2 + 32) >> 6;
        block[8 + j] = (e1 - e2 + 32) >> 6;
        block[12 + j] = (e0 - e3 + 32) >> 6;
    }
}

/// LevelScale factors from H.264 Table 8-13.
/// Indexed by [qp_rem][position_category] where position categories are:
/// 0: (0,0),(2,0),(0,2),(2,2)
/// 1: (1,1),(3,1),(1,3),(3,3)
/// 2: other positions
const LEVEL_SCALE: [[i32; 3]; 6] = [
    [10, 16, 13],
    [11, 18, 14],
    [13, 20, 16],
    [14, 23, 18],
    [16, 25, 20],
    [18, 29, 23],
];

/// Get the position category for a 4x4 block position (row, col).
fn position_category(row: usize, col: usize) -> usize {
    match (row % 2, col % 2) {
        (0, 0) => 0,
        (1, 1) => 1,
        _ => 2,
    }
}

/// Dequantize a 4x4 AC residual block in-place.
/// `scale` is the 4x4 scaling matrix (in scan order, default all-16 for flat scaling).
pub fn dequant_4x4(block: &mut [i32; 16], qp: i32, scale: &[u8; 16]) {
    let qp_per = qp / 6;
    let qp_rem = (qp % 6) as usize;

    for idx in 0..16 {
        if block[idx] != 0 {
            let (r, c) = ZIGZAG_4X4[idx];
            let v = LEVEL_SCALE[qp_rem][position_category(r, c)] * scale[idx] as i32;
            if qp_per >= 4 {
                block[idx] = (block[idx] * v) << (qp_per - 4);
            } else {
                block[idx] = (block[idx] * v + (1 << (3 - qp_per))) >> (4 - qp_per);
            }
        }
    }
}

/// Dequantize I16x16 luma DC coefficients after Hadamard.
/// Per spec 8.5.12.1, DC scaling uses scale[0] (the DC position of the scaling matrix).
pub fn dequant_luma_dc_i16x16(dc: &mut [i32; 16], qp: i32, scale_dc: u8) {
    let qp_per = qp / 6;
    let qp_rem = (qp % 6) as usize;
    let v = LEVEL_SCALE[qp_rem][0] * scale_dc as i32;

    if qp_per >= 6 {
        for d in dc.iter_mut() {
            *d = (*d * v) << (qp_per - 6);
        }
    } else {
        let round = 1 << (5 - qp_per);
        for d in dc.iter_mut() {
            *d = (*d * v + round) >> (6 - qp_per);
        }
    }
}

/// Dequantize chroma DC coefficients after Hadamard.
/// Per spec 8.5.12.2. Uses scale[0] from the chroma scaling matrix.
pub fn dequant_chroma_dc(dc: &mut [i32; 4], qp: i32, scale_dc: u8) {
    let qp_per = qp / 6;
    let qp_rem = (qp % 6) as usize;
    let v = LEVEL_SCALE[qp_rem][0] * scale_dc as i32;

    if qp_per >= 5 {
        for d in dc.iter_mut() {
            *d = (*d * v) << (qp_per - 5);
        }
    } else {
        let round = 1 << (4 - qp_per);
        for d in dc.iter_mut() {
            *d = (*d * v + round) >> (5 - qp_per);
        }
    }
}

/// QP_C lookup table from QP_I (H.264 Table 8-15).
/// qPI = clip3(0, 51, QP_Y + chroma_qp_index_offset)
/// QP_C = QPC_TABLE[qPI]
pub const QPC_TABLE: [i32; 52] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 29, 30, 31, 32, 32, 33, 34, 34, 35, 35,
    36, 36, 37, 37, 37, 38, 38, 38, 39, 39, 39, 39,
];

/// Compute chroma QP from luma QP and chroma_qp_index_offset.
pub fn chroma_qp(qp_y: i32, chroma_qp_index_offset: i32) -> i32 {
    let qpi = (qp_y + chroma_qp_index_offset).clamp(0, 51);
    QPC_TABLE[qpi as usize]
}

/// 8x8 zigzag scan for CAVLC (spec Table 8-12, rearranged for CAVLC 4-quad decode).
/// Structured as 4 groups of 16: each group is one 4x4 sub-block's scan positions
/// within the full 8x8 block. Value = row * 8 + col.
#[rustfmt::skip]
pub const ZIGZAG_8X8_CAVLC: [usize; 64] = [
    // 4 groups of 16: each maps CAVLC scan position → raster position in 8x8
     0,  9, 17, 18, 12, 40, 27,  7, 35, 57, 29, 30, 58, 38, 53, 47,
     1,  2, 24, 11, 19, 48, 20, 14, 42, 50, 22, 37, 59, 31, 60, 55,
     8,  3, 32,  4, 26, 41, 13, 21, 49, 43, 15, 44, 52, 39, 61, 62,
    16, 10, 25,  5, 33, 34,  6, 28, 56, 36, 23, 51, 45, 46, 54, 63,
];

/// 8x8 zigzag scan for CABAC (standard zigzag, value = row*8+col).
#[rustfmt::skip]
pub const ZIGZAG_8X8_CABAC: [usize; 64] = [
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
];

/// LevelScale factors for 8x8 blocks (H.264 Table 8-14).
/// Indexed by [qp_rem][position_category_8x8].
/// 8x8 blocks have 6 position categories (vs 3 for 4x4).
const LEVEL_SCALE_8X8: [[i32; 6]; 6] = [
    [20, 18, 32, 19, 25, 24],
    [22, 19, 35, 21, 28, 26],
    [26, 23, 42, 24, 33, 31],
    [28, 25, 45, 26, 35, 33],
    [32, 28, 51, 30, 40, 38],
    [36, 32, 58, 34, 46, 43],
];

/// Maps (row%4)*4 + (col%4) to one of 6 position categories for 8x8 dequant.
/// From H.264 spec Table 8-14.
const DEQUANT_8X8_POS_CAT: [usize; 16] = [
    0, 3, 4, 3, 3, 1, 5, 1, 4, 5, 2, 5, 3, 1, 5, 1,
];

/// Dequantize an 8x8 residual block in raster order.
/// `block[r*8+c]` contains coefficients in raster positions.
/// `scale` is the 8x8 scaling matrix (64 values in raster order).
pub fn dequant_8x8(block: &mut [i32; 64], qp: i32, scale: &[u8; 64]) {
    let qp_per = qp / 6;
    let qp_rem = (qp % 6) as usize;

    for (idx, coeff) in block.iter_mut().enumerate() {
        if *coeff != 0 {
            let r = idx / 8;
            let c = idx % 8;
            let cat = DEQUANT_8X8_POS_CAT[(r % 4) * 4 + (c % 4)];
            let v = LEVEL_SCALE_8X8[qp_rem][cat] * scale[r * 8 + c] as i32;
            if qp_per >= 6 {
                *coeff = (*coeff * v) << (qp_per - 6);
            } else {
                *coeff = (*coeff * v + (1 << (5 - qp_per))) >> (6 - qp_per);
            }
        }
    }
}

/// Inverse 8x8 integer DCT transform (H.264 spec 8.5.12).
/// Operates in-place on 64 coefficients in raster order (row-major, 8 per row).
/// Horizontal pass first (columns within each row), then vertical pass.
pub fn inverse_dct_8x8(block: &mut [i32; 64]) {
    // Add rounding bias to DC
    block[0] += 32;

    // Horizontal pass (column indices within each row)
    for i in 0..8 {
        let a0 = block[i] + block[i + 4 * 8];
        let a2 = block[i] - block[i + 4 * 8];
        let a4 = (block[i + 2 * 8] >> 1) - block[i + 6 * 8];
        let a6 = (block[i + 6 * 8] >> 1) + block[i + 2 * 8];

        let b0 = a0 + a6;
        let b2 = a2 + a4;
        let b4 = a2 - a4;
        let b6 = a0 - a6;

        let a1 = -block[i + 3 * 8] + block[i + 5 * 8]
            - block[i + 7 * 8] - (block[i + 7 * 8] >> 1);
        let a3 = block[i + 8] + block[i + 7 * 8]
            - block[i + 3 * 8] - (block[i + 3 * 8] >> 1);
        let a5 = -block[i + 8] + block[i + 7 * 8]
            + block[i + 5 * 8] + (block[i + 5 * 8] >> 1);
        let a7 = block[i + 3 * 8] + block[i + 5 * 8]
            + block[i + 8] + (block[i + 8] >> 1);

        let b1 = (a7 >> 2) + a1;
        let b3 = a3 + (a5 >> 2);
        let b5 = (a3 >> 2) - a5;
        let b7 = a7 - (a1 >> 2);

        block[i] = b0 + b7;
        block[i + 7 * 8] = b0 - b7;
        block[i + 8] = b2 + b5;
        block[i + 6 * 8] = b2 - b5;
        block[i + 2 * 8] = b4 + b3;
        block[i + 5 * 8] = b4 - b3;
        block[i + 3 * 8] = b6 + b1;
        block[i + 4 * 8] = b6 - b1;
    }

    // Vertical pass (row indices within each column), with >> 6 rounding
    for i in 0..8 {
        let a0 = block[i * 8] + block[4 + i * 8];
        let a2 = block[i * 8] - block[4 + i * 8];
        let a4 = (block[2 + i * 8] >> 1) - block[6 + i * 8];
        let a6 = (block[6 + i * 8] >> 1) + block[2 + i * 8];

        let b0 = a0 + a6;
        let b2 = a2 + a4;
        let b4 = a2 - a4;
        let b6 = a0 - a6;

        let a1 = -block[3 + i * 8] + block[5 + i * 8]
            - block[7 + i * 8] - (block[7 + i * 8] >> 1);
        let a3 = block[1 + i * 8] + block[7 + i * 8]
            - block[3 + i * 8] - (block[3 + i * 8] >> 1);
        let a5 = -block[1 + i * 8] + block[7 + i * 8]
            + block[5 + i * 8] + (block[5 + i * 8] >> 1);
        let a7 = block[3 + i * 8] + block[5 + i * 8]
            + block[1 + i * 8] + (block[1 + i * 8] >> 1);

        let b1 = (a7 >> 2) + a1;
        let b3 = a3 + (a5 >> 2);
        let b5 = (a3 >> 2) - a5;
        let b7 = a7 - (a1 >> 2);

        block[i * 8] = (b0 + b7) >> 6;
        block[1 + i * 8] = (b2 + b5) >> 6;
        block[2 + i * 8] = (b4 + b3) >> 6;
        block[3 + i * 8] = (b6 + b1) >> 6;
        block[4 + i * 8] = (b6 - b1) >> 6;
        block[5 + i * 8] = (b4 - b3) >> 6;
        block[6 + i * 8] = (b2 - b5) >> 6;
        block[7 + i * 8] = (b0 - b7) >> 6;
    }
}

/// Raster block index to (mb_row_offset, mb_col_offset) for luma 4x4 blocks.
/// Block ordering within a macroblock: inverse raster scan of 8x8 blocks,
/// then raster scan of 4x4 within each 8x8.
pub const BLOCK_INDEX_TO_OFFSET: [(usize, usize); 16] = [
    (0, 0), (0, 4), (4, 0), (4, 4),     // block 0-3 (top-left 8x8)
    (0, 8), (0, 12), (4, 8), (4, 12),    // block 4-7 (top-right 8x8)
    (8, 0), (8, 4), (12, 0), (12, 4),    // block 8-11 (bottom-left 8x8)
    (8, 8), (8, 12), (12, 8), (12, 12),  // block 12-15 (bottom-right 8x8)
];

/// coded_block_pattern mapping for I macroblocks (H.264 Table 9-4).
/// Index is the code_number from ue(v); value is the CBP.
/// Low 4 bits = luma CBP (one bit per 8x8 block), bits 4-5 = chroma CBP (0/1/2).
#[rustfmt::skip]
pub const CBP_INTRA_TABLE: [u8; 48] = [
    47, 31, 15,  0, 23, 27, 29, 30,  7, 11, 13, 14, 39, 43, 45, 46,
    16,  3,  5, 10, 12, 19, 21, 26, 28, 35, 37, 42, 44,  1,  2,  4,
     8, 17, 18, 20, 24,  6,  9, 22, 25, 32, 33, 34, 36, 40, 38, 41,
];

/// coded_block_pattern mapping for Inter macroblocks (H.264 Table 9-4b).
#[rustfmt::skip]
pub const CBP_INTER_TABLE: [u8; 48] = [
     0, 16,  1,  2,  4,  8, 32,  3,  5, 10, 12, 15, 47,  7, 11, 13,
    14,  6,  9, 31, 35, 37, 42, 44, 33, 34, 36, 40, 39, 43, 45, 46,
    17, 18, 20, 24, 19, 21, 26, 28, 23, 27, 29, 30, 22, 25, 38, 41,
];

/// Dequantize a full 4x4 block (including DC at position [0][0]) in raster order.
/// `scale` is the scaling matrix in scan order; mapped to raster via ZIGZAG_4X4.
pub fn dequant_4x4_full(block: &mut [i32; 16], qp: i32, scale: &[u8; 16]) {
    let qp_per = qp / 6;
    let qp_rem = (qp % 6) as usize;

    for r in 0..4 {
        for c in 0..4 {
            let idx = r * 4 + c;
            if block[idx] != 0 {
                let pc = position_category(r, c);
                // Find the scan-order index for this raster position to look up the scale
                let scan_idx = ZIGZAG_4X4.iter().position(|&(zr, zc)| zr == r && zc == c).unwrap();
                let v = LEVEL_SCALE[qp_rem][pc] * scale[scan_idx] as i32;
                if qp_per >= 4 {
                    block[idx] = (block[idx] * v) << (qp_per - 4);
                } else {
                    block[idx] = (block[idx] * v + (1 << (3 - qp_per))) >> (4 - qp_per);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inverse_hadamard_4x4_all_same() {
        // If all DC values are the same, Hadamard should concentrate energy in [0]
        let mut dc = [5i32; 16];
        inverse_hadamard_4x4(&mut dc);
        assert_eq!(dc[0], 80); // 5 * 16
        for &v in &dc[1..] {
            assert_eq!(v, 0);
        }
    }

    #[test]
    fn test_inverse_hadamard_2x2() {
        let mut dc = [1, 0, 0, 0];
        inverse_hadamard_2x2(&mut dc);
        assert_eq!(dc, [1, 1, 1, 1]);
    }
}
