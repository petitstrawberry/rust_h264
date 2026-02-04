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

    // Vertical pass (columns)
    for j in 0..4 {
        let z0 = tmp[j];
        let z1 = tmp[4 + j];
        let z2 = tmp[8 + j];
        let z3 = tmp[12 + j];

        let e0 = z0 + z2;
        let e1 = z0 - z2;
        let e2 = (z1 >> 1) - z3;
        let e3 = z1 + (z3 >> 1);

        // Output with rounding: (x + 32) >> 6
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
/// `qp` is the quantization parameter (QP_Y for luma, QP_C for chroma).
pub fn dequant_4x4(block: &mut [i32; 16], qp: i32) {
    let qp_per = qp / 6;
    let qp_rem = (qp % 6) as usize;

    for idx in 0..16 {
        if block[idx] != 0 {
            let (r, c) = ZIGZAG_4X4[idx];
            let v = LEVEL_SCALE[qp_rem][position_category(r, c)];
            if qp_per >= 0 {
                block[idx] = (block[idx] * v) << qp_per;
            } else {
                block[idx] = (block[idx] * v + (1 << (-qp_per - 1))) >> -qp_per;
            }
        }
    }
}

/// Dequantize I16x16 luma DC coefficients after Hadamard.
/// Per spec 8.5.12.1, the scaling is different from AC.
pub fn dequant_luma_dc_i16x16(dc: &mut [i32; 16], qp: i32) {
    let qp_per = qp / 6;
    let qp_rem = (qp % 6) as usize;
    let v = LEVEL_SCALE[qp_rem][0];

    if qp_per >= 2 {
        for d in dc.iter_mut() {
            *d = (*d * v) << (qp_per - 2);
        }
    } else {
        let round = 1 << (1 - qp_per);
        for d in dc.iter_mut() {
            *d = (*d * v + round) >> (2 - qp_per);
        }
    }
}

/// Dequantize chroma DC coefficients after Hadamard.
/// Per spec 8.5.12.2.
pub fn dequant_chroma_dc(dc: &mut [i32; 4], qp: i32) {
    let qp_per = qp / 6;
    let qp_rem = (qp % 6) as usize;
    let v = LEVEL_SCALE[qp_rem][0];

    if qp_per >= 1 {
        for d in dc.iter_mut() {
            *d = (*d * v) << (qp_per - 1);
        }
    } else {
        for d in dc.iter_mut() {
            *d = (*d * v) >> 1;
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

/// Dequantize a full 4x4 block (including DC at position [0][0]) in raster order.
pub fn dequant_4x4_full(block: &mut [i32; 16], qp: i32) {
    let qp_per = qp / 6;
    let qp_rem = (qp % 6) as usize;

    for r in 0..4 {
        for c in 0..4 {
            let idx = r * 4 + c;
            if block[idx] != 0 {
                let pc = match (r % 2, c % 2) {
                    (0, 0) => 0,
                    (1, 1) => 1,
                    _ => 2,
                };
                let v = LEVEL_SCALE[qp_rem][pc];
                block[idx] = (block[idx] * v) << qp_per;
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
