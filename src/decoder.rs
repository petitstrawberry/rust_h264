use std::collections::HashMap;

use crate::cavlc::parse_residual_block_cavlc;
use crate::intra_pred::{predict_chroma_8x8, predict_intra_16x16};
use crate::nal::{NalUnit, NalUnitType};
use crate::pps::{parse_pps, Pps};
use crate::residual::{
    chroma_qp, dequant_chroma_dc, dequant_luma_dc_i16x16, inverse_dct_4x4,
    inverse_hadamard_2x2, inverse_hadamard_4x4, BLOCK_INDEX_TO_OFFSET, ZIGZAG_4X4,
};
use crate::slice::{parse_slice_header, SliceType};
use crate::sps::{parse_sps, Sps};

/// A decoded YUV 4:2:0 frame.
#[derive(Debug)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
}

pub struct Decoder {
    sps_table: HashMap<u32, Sps>,
    pps_table: HashMap<u32, Pps>,
}

impl Decoder {
    pub fn new() -> Self {
        Self {
            sps_table: HashMap::new(),
            pps_table: HashMap::new(),
        }
    }

    /// Feed a NAL unit to the decoder. Returns a decoded frame if one is produced.
    pub fn decode_nal(&mut self, nal: &NalUnit) -> Result<Option<Frame>, &'static str> {
        match nal.nal_unit_type {
            NalUnitType::Sps => {
                let sps = parse_sps(&nal.rbsp)?;
                self.sps_table.insert(sps.seq_parameter_set_id, sps);
                Ok(None)
            }
            NalUnitType::Pps => {
                let pps = parse_pps(&nal.rbsp)?;
                self.pps_table.insert(pps.pic_parameter_set_id, pps);
                Ok(None)
            }
            NalUnitType::Sei => Ok(None),
            NalUnitType::SliceIdr | NalUnitType::Slice => {
                self.decode_slice(nal)
            }
            _ => Ok(None),
        }
    }

    fn decode_slice(&self, nal: &NalUnit) -> Result<Option<Frame>, &'static str> {
        // We need at least one SPS and PPS to decode.
        // Peek at PPS ID from the slice header to find the right one.
        // For now, try PPS 0 -> SPS 0 (the common case).
        let pps = self.pps_table.values().next().ok_or("no PPS available")?;
        let sps = self
            .sps_table
            .get(&pps.seq_parameter_set_id)
            .ok_or("no SPS available")?;

        let (header, mut reader) =
            parse_slice_header(&nal.rbsp, sps, pps, nal.nal_unit_type)?;

        if header.slice_type != SliceType::I {
            return Err("only I slices supported");
        }

        let width = sps.width();
        let height = sps.height();
        let mb_width = (width + 15) / 16;
        let mb_height = (height + 15) / 16;
        let total_mbs = (mb_width * mb_height) as usize;

        let mut frame = Frame {
            width,
            height,
            y: vec![0u8; (width * height) as usize],
            u: vec![0u8; (width * height / 4) as usize],
            v: vec![0u8; (width * height / 4) as usize],
        };

        let slice_qp = header.qp_y(pps);
        let mut prev_mb_qp = slice_qp;

        // nC tracking array: total_coeff for each 4x4 luma block
        let mut _nc_luma = vec![0u8; total_mbs * 16];

        for mb_idx in header.first_mb_in_slice as usize..total_mbs {
            let mb_x = (mb_idx % mb_width as usize) * 16;
            let mb_y = (mb_idx / mb_width as usize) * 16;

            let mb_type = reader.read_ue()?;

            if mb_type == 0 {
                // I_NxN (I4x4) — not yet supported
                return Err("I4x4 macroblocks not yet supported");
            }

            // I16x16 macroblock: mb_type 1-24
            if mb_type > 24 {
                return Err("unsupported mb_type for I slice");
            }

            let mt = mb_type - 1;
            let intra16x16_pred_mode = (mt % 4) as u8;
            let cbp_chroma = ((mt / 4) % 3) as u8;
            let cbp_luma = if mt >= 12 { 15u8 } else { 0u8 };

            // Read intra_chroma_pred_mode
            let intra_chroma_pred_mode = reader.read_ue()? as u8;

            // Read mb_qp_delta
            let mb_qp_delta = reader.read_se()?;
            let qp_y = ((prev_mb_qp + mb_qp_delta + 52) % 52 + 52) % 52;
            prev_mb_qp = qp_y;

            let qp_c = chroma_qp(qp_y, pps.chroma_qp_index_offset);

            // Parse luma DC (16 coefficients for I16x16)
            let mut luma_dc = [0i32; 16];

            parse_residual_block_cavlc(&mut reader, &mut luma_dc, 16, 0)?;



            // Parse luma AC (16 blocks of 15 coefficients each, in scan order)
            // luma_ac_scan[blk][0..14] = AC coefficients in zigzag scan positions 1-15
            let mut luma_ac_scan = [[0i32; 15]; 16];
            if cbp_luma != 0 {
                for blk in 0..16 {
                    parse_residual_block_cavlc(&mut reader, &mut luma_ac_scan[blk], 15, 0)?;
                }
            }

            // Parse chroma DC
            let mut chroma_dc_cb = [0i32; 4];
            let mut chroma_dc_cr = [0i32; 4];
            if cbp_chroma >= 1 {
                parse_residual_block_cavlc(&mut reader, &mut chroma_dc_cb, 4, -1)?;
                parse_residual_block_cavlc(&mut reader, &mut chroma_dc_cr, 4, -1)?;
            }

            // Parse chroma AC (in zigzag scan positions 1-15)
            let mut chroma_ac_scan_cb = [[0i32; 15]; 4];
            let mut chroma_ac_scan_cr = [[0i32; 15]; 4];
            if cbp_chroma >= 2 {
                for blk in 0..4 {
                    parse_residual_block_cavlc(&mut reader, &mut chroma_ac_scan_cb[blk], 15, 0)?;
                }
                for blk in 0..4 {
                    parse_residual_block_cavlc(&mut reader, &mut chroma_ac_scan_cr[blk], 15, 0)?;
                }
            }

            // === Reconstruct luma ===

            // Unzigzag DC coefficients to raster order, then Hadamard, then dequant
            let mut luma_dc_raster = [0i32; 16];
            for i in 0..16 {
                let (r, c) = ZIGZAG_4X4[i];
                luma_dc_raster[r * 4 + c] = luma_dc[i];
            }

            inverse_hadamard_4x4(&mut luma_dc_raster);

            dequant_luma_dc_i16x16(&mut luma_dc_raster, qp_y);


            // DC raster index to MB block index mapping:
            // DC raster (row, col) -> block index via the inverse of BLOCK_INDEX_TO_OFFSET
            // DC raster[r*4+c] corresponds to the 4x4 block at MB position (r*4, c*4)
            const DC_RASTER_TO_BLOCK: [usize; 16] = [
                0, 1, 4, 5,
                2, 3, 6, 7,
                8, 9, 12, 13,
                10, 11, 14, 15,
            ];

            // For each 4x4 block: place DC, unzigzag AC, dequant AC, inverse DCT
            let mut luma_residual = [0i32; 256]; // 16x16
            for blk in 0..16 {
                let (blk_row, blk_col) = BLOCK_INDEX_TO_OFFSET[blk];

                // Build 4x4 block in raster order
                let mut block_raster = [0i32; 16];

                // DC from Hadamard output: find which DC raster index maps to this block
                let dc_idx = DC_RASTER_TO_BLOCK.iter().position(|&b| b == blk).unwrap();
                block_raster[0] = luma_dc_raster[dc_idx];

                // Unzigzag AC: scan positions 1-15 map to raster positions via ZIGZAG_4X4
                if cbp_luma != 0 {
                    for scan_idx in 0..15 {
                        let (r, c) = ZIGZAG_4X4[scan_idx + 1]; // scan positions 1-15
                        block_raster[r * 4 + c] = luma_ac_scan[blk][scan_idx];
                    }
                    dequant_4x4_ac_raster(&mut block_raster, qp_y);
                }

                inverse_dct_4x4(&mut block_raster);

                // Write to residual buffer
                for r in 0..4 {
                    for c in 0..4 {
                        luma_residual[(blk_row + r) * 16 + blk_col + c] = block_raster[r * 4 + c];
                    }
                }
            }

            // Generate I16x16 prediction
            let mut luma_pred = [0u8; 256];
            // For the first MB at (0,0), no neighbors
            let above: Option<Vec<u8>> = if mb_y > 0 {
                Some(
                    (0..16)
                        .map(|x| frame.y[(mb_y - 1) * width as usize + mb_x + x])
                        .collect(),
                )
            } else {
                None
            };
            let left: Option<Vec<u8>> = if mb_x > 0 {
                Some(
                    (0..16)
                        .map(|y| frame.y[(mb_y + y) * width as usize + mb_x - 1])
                        .collect(),
                )
            } else {
                None
            };
            let above_left = if mb_x > 0 && mb_y > 0 {
                Some(frame.y[(mb_y - 1) * width as usize + mb_x - 1])
            } else {
                None
            };

            predict_intra_16x16(
                intra16x16_pred_mode,
                above.as_deref(),
                left.as_deref(),
                above_left,
                &mut luma_pred,
            );

            // Add residual to prediction and write to frame
            for y in 0..16 {
                for x in 0..16 {
                    let pred = luma_pred[y * 16 + x] as i32;
                    let res = luma_residual[y * 16 + x];
                    let val = (pred + res).clamp(0, 255) as u8;
                    frame.y[(mb_y + y) * width as usize + mb_x + x] = val;
                }
            }

            // === Reconstruct chroma ===
            let chroma_width = (width / 2) as usize;
            let chroma_mb_x = mb_x / 2;
            let chroma_mb_y = mb_y / 2;

            for (plane_dc, plane_ac_scan, plane_buf) in [
                (&mut chroma_dc_cb, &chroma_ac_scan_cb, &mut frame.u),
                (&mut chroma_dc_cr, &chroma_ac_scan_cr, &mut frame.v),
            ] {
                // Inverse Hadamard then dequant on chroma DC
                if cbp_chroma >= 1 {
                    inverse_hadamard_2x2(plane_dc);
                    dequant_chroma_dc(plane_dc, qp_c);
                }

                // Reconstruct each 4x4 chroma block
                let mut chroma_residual = [0i32; 64]; // 8x8
                for blk in 0..4 {
                    let blk_row = (blk / 2) * 4;
                    let blk_col = (blk % 2) * 4;

                    // Build 4x4 block in raster order
                    let mut block_raster = [0i32; 16];
                    block_raster[0] = plane_dc[blk];

                    if cbp_chroma >= 2 {
                        for scan_idx in 0..15 {
                            let (r, c) = ZIGZAG_4X4[scan_idx + 1];
                            block_raster[r * 4 + c] = plane_ac_scan[blk][scan_idx];
                        }
                        dequant_4x4_ac_raster(&mut block_raster, qp_c);
                    }

                    inverse_dct_4x4(&mut block_raster);

                    for r in 0..4 {
                        for c in 0..4 {
                            chroma_residual[(blk_row + r) * 8 + blk_col + c] = block_raster[r * 4 + c];
                        }
                    }
                }

                // Chroma prediction
                let mut chroma_pred = [0u8; 64];
                let above_c: Option<Vec<u8>> = if chroma_mb_y > 0 {
                    Some(
                        (0..8)
                            .map(|x| {
                                plane_buf[(chroma_mb_y - 1) * chroma_width + chroma_mb_x + x]
                            })
                            .collect(),
                    )
                } else {
                    None
                };
                let left_c: Option<Vec<u8>> = if chroma_mb_x > 0 {
                    Some(
                        (0..8)
                            .map(|y| {
                                plane_buf[(chroma_mb_y + y) * chroma_width + chroma_mb_x - 1]
                            })
                            .collect(),
                    )
                } else {
                    None
                };
                let above_left_c = if chroma_mb_x > 0 && chroma_mb_y > 0 {
                    Some(
                        plane_buf[(chroma_mb_y - 1) * chroma_width + chroma_mb_x - 1],
                    )
                } else {
                    None
                };

                predict_chroma_8x8(
                    intra_chroma_pred_mode,
                    above_c.as_deref(),
                    left_c.as_deref(),
                    above_left_c,
                    &mut chroma_pred,
                );

                // Add residual and write
                for y in 0..8 {
                    for x in 0..8 {
                        let pred = chroma_pred[y * 8 + x] as i32;
                        let res = chroma_residual[y * 8 + x];
                        let val = (pred + res).clamp(0, 255) as u8;
                        plane_buf[(chroma_mb_y + y) * chroma_width + chroma_mb_x + x] = val;
                    }
                }
            }
        }

        Ok(Some(frame))
    }
}

/// Dequantize AC coefficients in a 4x4 block in raster order (skip position [0][0] which is DC).
fn dequant_4x4_ac_raster(block: &mut [i32; 16], qp: i32) {
    let qp_per = qp / 6;
    let qp_rem = (qp % 6) as usize;

    const LEVEL_SCALE: [[i32; 3]; 6] = [
        [10, 16, 13],
        [11, 18, 14],
        [13, 20, 16],
        [14, 23, 18],
        [16, 25, 20],
        [18, 29, 23],
    ];

    for r in 0..4 {
        for c in 0..4 {
            if r == 0 && c == 0 {
                continue; // DC already scaled
            }
            let idx = r * 4 + c;
            if block[idx] != 0 {
                let pc = match (r % 2, c % 2) {
                    (0, 0) => 0,
                    (1, 1) => 1,
                    _ => 2,
                };
                let v = LEVEL_SCALE[qp_rem][pc];
                block[idx] = block[idx] * v << qp_per;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nal::parse_annex_b;

    #[test]
    fn test_decode_single_idr_frame() {
        let h264_data = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/single_frame.h264"
        ))
        .unwrap();
        let expected_yuv = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/single_frame.yuv"
        ))
        .unwrap();

        let nals = parse_annex_b(&h264_data);
        let mut decoder = Decoder::new();
        let mut frame = None;
        for nal in &nals {
            if let Some(f) = decoder.decode_nal(nal).unwrap() {
                frame = Some(f);
            }
        }
        let frame = frame.expect("should have decoded a frame");

        assert_eq!(frame.width, 16);
        assert_eq!(frame.height, 16);

        let mut output = Vec::new();
        output.extend_from_slice(&frame.y);
        output.extend_from_slice(&frame.u);
        output.extend_from_slice(&frame.v);
        assert_eq!(output.len(), expected_yuv.len());
        assert_eq!(output, expected_yuv);
    }
}
