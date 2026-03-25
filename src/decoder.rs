use std::collections::HashMap;

use std::rc::Rc;

use crate::bitstream::BitstreamReader;
use crate::cavlc::parse_residual_block_cavlc;
use crate::dpb::{DecodedPicture, Dpb, ReferenceStatus};
use crate::error::DecodeError;
use crate::inter_pred;
use crate::deblock::{self, MbInfo, MbType};
use crate::intra_pred::{predict_chroma_8x8, predict_intra_16x16, predict_intra_4x4};
use crate::nal::{NalUnit, NalUnitType};
use crate::pps::{parse_pps, Pps};
use crate::residual::{
    chroma_qp, dequant_4x4_full, dequant_chroma_dc, dequant_luma_dc_i16x16, inverse_dct_4x4,
    inverse_hadamard_2x2, inverse_hadamard_4x4, BLOCK_INDEX_TO_OFFSET, CBP_INTER_TABLE,
    CBP_INTRA_TABLE, ZIGZAG_4X4,
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
    dpb: Dpb,
}

impl Default for Decoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Decoder {
    pub fn new() -> Self {
        Self {
            sps_table: HashMap::new(),
            pps_table: HashMap::new(),
            dpb: Dpb::new(0),
        }
    }

    /// Feed a NAL unit to the decoder. Returns a decoded frame if one is produced.
    pub fn decode_nal(&mut self, nal: &NalUnit) -> Result<Option<Frame>, DecodeError> {
        match nal.nal_unit_type {
            NalUnitType::Sps => {
                let sps = parse_sps(&nal.rbsp)?;
                self.dpb.set_max_ref_frames(sps.max_num_ref_frames);
                self.sps_table.insert(sps.seq_parameter_set_id, sps);
                Ok(None)
            }
            NalUnitType::Pps => {
                let pps_id_sps = {
                    // Peek at seq_parameter_set_id to find the right SPS
                    let mut peek = BitstreamReader::new(&nal.rbsp);
                    let _ = peek.read_ue(); // pic_parameter_set_id
                    peek.read_ue().ok()
                };
                let sps_ref = pps_id_sps.and_then(|id| self.sps_table.get(&id));
                let pps = parse_pps(&nal.rbsp, sps_ref)?;
                self.pps_table.insert(pps.pic_parameter_set_id, pps);
                Ok(None)
            }
            NalUnitType::Sei => Ok(None),
            NalUnitType::SliceIdr | NalUnitType::Slice => self.decode_slice(nal),
            _ => Ok(None),
        }
    }

    fn decode_slice(&mut self, nal: &NalUnit) -> Result<Option<Frame>, DecodeError> {
        let pps = self.pps_table.values().next().ok_or(DecodeError::InvalidSyntax("no PPS available"))?;
        let sps = self
            .sps_table
            .get(&pps.seq_parameter_set_id)
            .ok_or(DecodeError::InvalidSyntax("no SPS available"))?;

        let (header, mut reader) =
            parse_slice_header(&nal.rbsp, sps, pps, nal.nal_unit_type, nal.nal_ref_idc)?;

        if header.slice_type != SliceType::I && header.slice_type != SliceType::P {
            return Err(DecodeError::from("only I and P slices supported"));
        }
        let is_p_slice = header.slice_type == SliceType::P;

        // Build reference picture list for P slices
        let ref_pic_list = if is_p_slice {
            self.dpb.short_term_ref_list()
        } else {
            vec![]
        };

        let width = sps.width();
        let height = sps.height();
        let mb_width = width.div_ceil(16);
        let mb_height = height.div_ceil(16);
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

        // nC tracking arrays: total_coeff for each 4x4 block
        let mut nc_luma = vec![0u8; total_mbs * 16];
        let mut nc_cb = vec![0u8; total_mbs * 4];
        let mut nc_cr = vec![0u8; total_mbs * 4];

        // I4x4 prediction mode storage (for neighbor prediction mode derivation).
        // Default to DC (2): per H.264 spec 8.3.1.1, non-I4x4 neighbors (I16x16, I_PCM)
        // use inferred mode DC for prediction mode derivation.
        let mut i4x4_modes = vec![2u8; total_mbs * 16];

        // Motion vector and reference index storage (per 4x4 block)
        let mut mv_store = vec![[0i16; 2]; total_mbs * 16];
        let mut ref_idx_store = vec![-1i8; total_mbs * 16];

        // Per-MB metadata for the deblocking filter
        let mut mb_info = vec![
            MbInfo {
                mb_type: MbType::Intra,
                qp_y: slice_qp,
            };
            total_mbs
        ];

        let mut mb_skip_run: i32 = -1; // -1 = not initialized for P slices

        let mut mb_idx = header.first_mb_in_slice as usize;
        while mb_idx < total_mbs {
            let mb_x = (mb_idx % mb_width as usize) * 16;
            let mb_y = (mb_idx / mb_width as usize) * 16;
            let stride = width as usize;

            // P-slice skip run handling
            if is_p_slice {
                if mb_skip_run < 0 {
                    mb_skip_run = reader.read_ue()? as i32;
                }
                if mb_skip_run > 0 {
                    mb_skip_run -= 1;
                    // P_Skip: MV = median predictor, ref_idx = 0, no residual
                    let (mvp_x, mvp_y) = predict_mv_skip(
                        &mv_store, &ref_idx_store, mb_idx, mb_width as usize,
                    );
                    if let Some(ref_pic) = ref_pic_list.first() {
                        let mut luma_pred = [0u8; 256];
                        inter_pred::luma_mc(
                            ref_pic, mb_x as i32, mb_y as i32,
                            mvp_x as i32, mvp_y as i32, 16, 16, &mut luma_pred,
                        );
                        for r in 0..16 {
                            for c in 0..16 {
                                frame.y[(mb_y + r) * stride + mb_x + c] = luma_pred[r * 16 + c];
                            }
                        }
                        let cw = (width / 2) as usize;
                        let cx = mb_x / 2;
                        let cy = mb_y / 2;
                        let mut cb_pred = [0u8; 64];
                        let mut cr_pred = [0u8; 64];
                        inter_pred::chroma_mc(
                            &ref_pic.u, cw, (height / 2) as usize,
                            cx as i32, cy as i32, mvp_x as i32, mvp_y as i32,
                            8, 8, &mut cb_pred,
                        );
                        inter_pred::chroma_mc(
                            &ref_pic.v, cw, (height / 2) as usize,
                            cx as i32, cy as i32, mvp_x as i32, mvp_y as i32,
                            8, 8, &mut cr_pred,
                        );
                        for r in 0..8 {
                            for c in 0..8 {
                                frame.u[(cy + r) * cw + cx + c] = cb_pred[r * 8 + c];
                                frame.v[(cy + r) * cw + cx + c] = cr_pred[r * 8 + c];
                            }
                        }
                    }
                    // Store MV and ref for neighbors
                    for blk in 0..16 {
                        mv_store[mb_idx * 16 + blk] = [mvp_x, mvp_y];
                        ref_idx_store[mb_idx * 16 + blk] = 0;
                    }
                    mb_info[mb_idx] = MbInfo {
                        mb_type: MbType::Inter,
                        qp_y: prev_mb_qp,
                    };
                    mb_idx += 1;
                    continue;
                }
                // mb_skip_run == 0: parse the next MB normally
                mb_skip_run = -1; // reset for next iteration
            }

            let raw_mb_type = reader.read_ue()?;
            // For P slices, mb_type >= 5 means intra (subtract 5)
            let (mb_type, is_inter) = if is_p_slice && raw_mb_type < 5 {
                (raw_mb_type, true)
            } else {
                let itype = if is_p_slice { raw_mb_type - 5 } else { raw_mb_type };
                (itype, false)
            };

            if is_inter {
                // === Inter (P) macroblock ===
                let (part_w, part_h, num_parts) = match mb_type {
                    0 => (16usize, 16usize, 1usize), // P_L0_16x16
                    1 => (16, 8, 2),                   // P_L0_L0_16x8
                    2 => (8, 16, 2),                   // P_L0_L0_8x16
                    _ => return Err(DecodeError::from("unsupported P-slice mb_type (P_8x8)")),
                };

                // Parse ref_idx for each partition
                let mut part_ref = [0i8; 2];
                for ref_entry in part_ref.iter_mut().take(num_parts) {
                    if header.num_ref_idx_l0_active > 1 {
                        *ref_entry = reader.read_te(header.num_ref_idx_l0_active - 1)? as i8;
                    }
                }

                // Parse MVD and compute final MV for each partition
                // Parse MVD and compute final MV for each partition.
                // Store each partition's MV immediately so the next partition's
                // predictor can read it (partition 1's above neighbor is partition 0).
                let mut part_mv = [[0i16; 2]; 2];
                for p in 0..num_parts {
                    let mvd_x = reader.read_se()? as i16;
                    let mvd_y = reader.read_se()? as i16;
                    let (mvp_x, mvp_y) = predict_mv(
                        &mv_store, &ref_idx_store, mb_idx, mb_width as usize,
                        p, part_w, part_h, part_ref[p],
                    );
                    part_mv[p] = [mvp_x + mvd_x, mvp_y + mvd_y];

                    // Store MV/ref immediately for this partition's 4x4 blocks
                    let (py_off, px_off) = match mb_type {
                        1 => (p * 8, 0),  // 16x8
                        2 => (0, p * 8),  // 8x16
                        _ => (0, 0),
                    };
                    for r in (0..part_h).step_by(4) {
                        for c in (0..part_w).step_by(4) {
                            let lr = (py_off + r) / 4;
                            let lc = (px_off + c) / 4;
                            if let Some(blk) = BLOCK_INDEX_TO_OFFSET
                                .iter()
                                .position(|&(br, bc)| br / 4 == lr && bc / 4 == lc)
                            {
                                mv_store[mb_idx * 16 + blk] = part_mv[p];
                                ref_idx_store[mb_idx * 16 + blk] = part_ref[p];
                            }
                        }
                    }
                }

                // Parse CBP using inter table
                let cbp_code = reader.read_ue()? as usize;
                if cbp_code >= 48 {
                    return Err(DecodeError::from("invalid coded_block_pattern"));
                }
                let cbp = CBP_INTER_TABLE[cbp_code];
                let cbp_luma = cbp & 0x0F;
                let cbp_chroma = cbp >> 4;

                let qp_y = if cbp_luma != 0 || cbp_chroma != 0 {
                    let mb_qp_delta = reader.read_se()?;
                    ((prev_mb_qp + mb_qp_delta + 52) % 52 + 52) % 52
                } else {
                    prev_mb_qp
                };
                prev_mb_qp = qp_y;
                let qp_c = chroma_qp(qp_y, pps.chroma_qp_index_offset);

                // Decode residual for each partition
                // Parse luma residual blocks
                let mut luma_residual = [0i32; 256];
                for blk in 0..16 {
                    if cbp_luma & (1 << (blk / 4)) != 0 {
                        let nc = compute_nc(&nc_luma, mb_idx, mb_width as usize, blk, 16);
                        let mut block_coeffs = [0i32; 16];
                        let tc = parse_residual_block_cavlc(
                            &mut reader, &mut block_coeffs, 16, nc,
                        )?;
                        nc_luma[mb_idx * 16 + blk] = tc;

                        let mut raster = [0i32; 16];
                        for i in 0..16 {
                            let (r, c) = ZIGZAG_4X4[i];
                            raster[r * 4 + c] = block_coeffs[i];
                        }
                        dequant_4x4_full(&mut raster, qp_y, &pps.scaling_list_4x4[3]);
                        inverse_dct_4x4(&mut raster);

                        let (blk_row, blk_col) = BLOCK_INDEX_TO_OFFSET[blk];
                        for r in 0..4 {
                            for c in 0..4 {
                                luma_residual[(blk_row + r) * 16 + blk_col + c] = raster[r * 4 + c];
                            }
                        }
                    }
                }

                // Motion compensate and add residual for each partition
                for p in 0..num_parts {
                    let (py_off, px_off) = match mb_type {
                        1 => (p * 8, 0),
                        2 => (0, p * 8),
                        _ => (0, 0),
                    };
                    let ref_pic = &ref_pic_list[part_ref[p] as usize];
                    let mut luma_pred = vec![0u8; part_w * part_h];
                    inter_pred::luma_mc(
                        ref_pic,
                        (mb_x + px_off) as i32, (mb_y + py_off) as i32,
                        part_mv[p][0] as i32, part_mv[p][1] as i32,
                        part_w, part_h, &mut luma_pred,
                    );
                    for r in 0..part_h {
                        for c in 0..part_w {
                            let val = (luma_pred[r * part_w + c] as i32
                                + luma_residual[(py_off + r) * 16 + px_off + c])
                                .clamp(0, 255) as u8;
                            frame.y[(mb_y + py_off + r) * stride + mb_x + px_off + c] = val;
                        }
                    }
                }

                // Chroma
                let mut chroma_dc_cb = [0i32; 4];
                let mut chroma_dc_cr = [0i32; 4];
                if cbp_chroma >= 1 {
                    parse_residual_block_cavlc(&mut reader, &mut chroma_dc_cb, 4, -1)?;
                    parse_residual_block_cavlc(&mut reader, &mut chroma_dc_cr, 4, -1)?;
                }
                let mut chroma_ac_scan_cb = [[0i32; 15]; 4];
                let mut chroma_ac_scan_cr = [[0i32; 15]; 4];
                if cbp_chroma >= 2 {
                    for blk in 0..4 {
                        let nc = compute_nc(&nc_cb, mb_idx, mb_width as usize, blk, 4);
                        let tc = parse_residual_block_cavlc(
                            &mut reader, &mut chroma_ac_scan_cb[blk], 15, nc,
                        )?;
                        nc_cb[mb_idx * 4 + blk] = tc;
                    }
                    for blk in 0..4 {
                        let nc = compute_nc(&nc_cr, mb_idx, mb_width as usize, blk, 4);
                        let tc = parse_residual_block_cavlc(
                            &mut reader, &mut chroma_ac_scan_cr[blk], 15, nc,
                        )?;
                        nc_cr[mb_idx * 4 + blk] = tc;
                    }
                }

                let chroma_width = (width / 2) as usize;
                let chroma_mb_x = mb_x / 2;
                let chroma_mb_y = mb_y / 2;

                // Chroma MC + residual for each chroma plane
                // Each partition gets its own chroma MC with its own MV
                {
                    for (plane_dc, plane_ac, frame_plane, scale_idx) in [
                        (&mut chroma_dc_cb, &chroma_ac_scan_cb, &mut frame.u, 4usize),
                        (&mut chroma_dc_cr, &chroma_ac_scan_cr, &mut frame.v, 5usize),
                    ] {
                        let chroma_scale = &pps.scaling_list_4x4[scale_idx];
                        if cbp_chroma >= 1 {
                            inverse_hadamard_2x2(plane_dc);
                            dequant_chroma_dc(plane_dc, qp_c, chroma_scale[0]);
                        }

                        let mut chroma_residual = [0i32; 64];
                        for blk in 0..4 {
                            let blk_row = (blk / 2) * 4;
                            let blk_col = (blk % 2) * 4;
                            let mut block_raster = [0i32; 16];
                            block_raster[0] = plane_dc[blk];
                            if cbp_chroma >= 2 {
                                for scan_idx in 0..15 {
                                    let (r, c) = ZIGZAG_4X4[scan_idx + 1];
                                    block_raster[r * 4 + c] = plane_ac[blk][scan_idx];
                                }
                                dequant_4x4_ac_raster(&mut block_raster, qp_c, chroma_scale);
                            }
                            inverse_dct_4x4(&mut block_raster);
                            for r in 0..4 {
                                for c in 0..4 {
                                    chroma_residual[(blk_row + r) * 8 + blk_col + c] =
                                        block_raster[r * 4 + c];
                                }
                            }
                        }

                        // MC each partition's chroma region separately
                        let mut chroma_pred = [0u8; 64];
                        for p in 0..num_parts {
                            let (cy_off, cx_off, cw, ch) = match mb_type {
                                1 => (p * 4, 0, 8, 4),   // 16x8 → chroma 8x4 per partition
                                2 => (0, p * 4, 4, 8),   // 8x16 → chroma 4x8 per partition
                                _ => (0, 0, 8, 8),       // 16x16
                            };
                            let part_ref_pic = &ref_pic_list[part_ref[p] as usize];
                            let chroma_ref = if scale_idx == 4 {
                                &part_ref_pic.u
                            } else {
                                &part_ref_pic.v
                            };
                            let mut part_pred = vec![0u8; cw * ch];
                            inter_pred::chroma_mc(
                                chroma_ref, chroma_width, (height / 2) as usize,
                                (chroma_mb_x + cx_off) as i32,
                                (chroma_mb_y + cy_off) as i32,
                                part_mv[p][0] as i32, part_mv[p][1] as i32,
                                cw, ch, &mut part_pred,
                            );
                            for r in 0..ch {
                                for c in 0..cw {
                                    chroma_pred[(cy_off + r) * 8 + cx_off + c] = part_pred[r * cw + c];
                                }
                            }
                        }

                        for y in 0..8 {
                            for x in 0..8 {
                                let val = (chroma_pred[y * 8 + x] as i32
                                    + chroma_residual[y * 8 + x])
                                    .clamp(0, 255) as u8;
                                frame_plane[(chroma_mb_y + y) * chroma_width + chroma_mb_x + x] = val;
                            }
                        }
                    }
                }

                mb_info[mb_idx] = MbInfo {
                    mb_type: MbType::Inter,
                    qp_y,
                };
                mb_idx += 1;
                continue;
            }

            // === Intra macroblock (I4x4, I16x16, I_PCM) ===
            // Variables shared between I4x4/I16x16 for chroma reconstruction
            let intra_chroma_pred_mode;
            let cbp_chroma: u8;
            let qp_y;
            let qp_c;

            if mb_type == 0 {
                // === I_NxN (I4x4) macroblock ===

                // Parse 16 I4x4 prediction modes
                let mut pred_modes = [2u8; 16];
                for blk in 0..16 {
                    let prev_flag = reader.read_bit()?;
                    let predicted = predict_i4x4_mode(
                        &i4x4_modes, mb_idx, mb_width as usize, blk,
                    );
                    if prev_flag != 0 {
                        pred_modes[blk] = predicted;
                    } else {
                        let rem = reader.read_bits(3)? as u8;
                        pred_modes[blk] = if rem < predicted { rem } else { rem + 1 };
                    }
                    i4x4_modes[mb_idx * 16 + blk] = pred_modes[blk];
                }
                intra_chroma_pred_mode = reader.read_ue()? as u8;

                let cbp_code = reader.read_ue()? as usize;
                if cbp_code >= 48 {
                    return Err(DecodeError::from("invalid coded_block_pattern"));
                }
                let cbp = CBP_INTRA_TABLE[cbp_code];
                let cbp_luma = cbp & 0x0F;
                cbp_chroma = cbp >> 4;

                if cbp_luma != 0 || cbp_chroma != 0 {
                    let mb_qp_delta = reader.read_se()?;
                    qp_y = ((prev_mb_qp + mb_qp_delta + 52) % 52 + 52) % 52;
                } else {
                    qp_y = prev_mb_qp;
                }
                prev_mb_qp = qp_y;
                qp_c = chroma_qp(qp_y, pps.chroma_qp_index_offset);

                // Parse and reconstruct each 4x4 luma block sequentially
                for blk in 0..16 {
                    let (blk_row, blk_col) = BLOCK_INDEX_TO_OFFSET[blk];
                    let px = mb_x + blk_col;
                    let py = mb_y + blk_row;

                    // Parse residual
                    let mut block_coeffs = [0i32; 16];
                    if cbp_luma & (1 << (blk / 4)) != 0 {
                        let nc = compute_nc(&nc_luma, mb_idx, mb_width as usize, blk, 16);
                        let tc = parse_residual_block_cavlc(
                            &mut reader, &mut block_coeffs, 16, nc,
                        )?;
                        nc_luma[mb_idx * 16 + blk] = tc;

                        // Unzigzag: convert from zigzag scan order to raster order
                        let mut raster = [0i32; 16];
                        for i in 0..16 {
                            let (r, c) = ZIGZAG_4X4[i];
                            raster[r * 4 + c] = block_coeffs[i];
                        }
                        block_coeffs = raster;

                        dequant_4x4_full(&mut block_coeffs, qp_y, &pps.scaling_list_4x4[0]);
                    }
                    inverse_dct_4x4(&mut block_coeffs);
                    // Gather neighbor samples for I4x4 prediction
                    let above_buf: Option<[u8; 8]> = if py > 0 {
                        let mut buf = [0u8; 8];
                        // Read 4 above pixels
                        for (i, b) in buf.iter_mut().enumerate().take(4) {
                            *b = frame.y[(py - 1) * stride + px + i];
                        }
                        // Above-right pixels (4 more): available only if the 4x4 block
                        // containing those pixels has already been decoded.
                        // Per H.264 spec 6.4.12, blocks 3,7,11,13,15 within the MB
                        // have above-right unavailable (the source block is decoded later).
                        // Also unavailable if at the right edge of the picture, or at the
                        // right edge of the MB when above is within the current MB.
                        let local_row = py - mb_y;
                        let topright_avail = if local_row == 0 {
                            // Above row is in the MB above (fully decoded).
                            // Above-right is available unless beyond picture width.
                            px + 4 < stride
                        } else {
                            // Above row is within current MB; above-right block may
                            // not be decoded yet.
                            !matches!(blk, 3 | 7 | 11 | 13 | 15)
                        };
                        if topright_avail {
                            for (i, b) in buf.iter_mut().enumerate().skip(4) {
                                let col = (px + i).min(stride - 1);
                                *b = frame.y[(py - 1) * stride + col];
                            }
                        } else {
                            // Replicate the last above pixel (spec 8.3.1.2.1)
                            let last = buf[3];
                            buf[4..8].fill(last);
                        }
                        Some(buf)
                    } else {
                        None
                    };
                    let left_buf: Option<[u8; 4]> = if px > 0 {
                        let mut buf = [0u8; 4];
                        for (i, b) in buf.iter_mut().enumerate() {
                            *b = frame.y[(py + i) * stride + px - 1];
                        }
                        Some(buf)
                    } else {
                        None
                    };
                    let above_left_val = if px > 0 && py > 0 {
                        Some(frame.y[(py - 1) * stride + px - 1])
                    } else {
                        None
                    };

                    let mut pred = [0u8; 16];
                    predict_intra_4x4(
                        pred_modes[blk],
                        above_buf.as_ref().map(|b| &b[..]),
                        left_buf.as_ref().map(|b| &b[..]),
                        above_left_val,
                        &mut pred,
                    );

                    // Add residual and write to frame
                    for r in 0..4 {
                        for c in 0..4 {
                            let val = (pred[r * 4 + c] as i32 + block_coeffs[r * 4 + c])
                                .clamp(0, 255) as u8;
                            frame.y[(py + r) * stride + px + c] = val;
                        }
                    }
                }
            } else if mb_type <= 24 {
                // === I16x16 macroblock ===
                let mt = mb_type - 1;
                let intra16x16_pred_mode = (mt % 4) as u8;
                cbp_chroma = ((mt / 4) % 3) as u8;
                let cbp_luma = if mt >= 12 { 15u8 } else { 0u8 };

                intra_chroma_pred_mode = reader.read_ue()? as u8;

                let mb_qp_delta = reader.read_se()?;
                qp_y = ((prev_mb_qp + mb_qp_delta + 52) % 52 + 52) % 52;
                prev_mb_qp = qp_y;
                qp_c = chroma_qp(qp_y, pps.chroma_qp_index_offset);

                // Parse luma DC
                let mut luma_dc = [0i32; 16];
                let nc_dc = compute_nc(&nc_luma, mb_idx, mb_width as usize, 0, 16);
                parse_residual_block_cavlc(&mut reader, &mut luma_dc, 16, nc_dc)?;

                // Parse luma AC
                let mut luma_ac_scan = [[0i32; 15]; 16];
                if cbp_luma != 0 {
                    for blk in 0..16 {
                        let nc =
                            compute_nc(&nc_luma, mb_idx, mb_width as usize, blk, 16);
                        let tc = parse_residual_block_cavlc(
                            &mut reader,
                            &mut luma_ac_scan[blk],
                            15,
                            nc,
                        )?;
                        nc_luma[mb_idx * 16 + blk] = tc;
                    }
                }

                // Unzigzag DC, Hadamard, dequant
                let mut luma_dc_raster = [0i32; 16];
                for i in 0..16 {
                    let (r, c) = ZIGZAG_4X4[i];
                    luma_dc_raster[r * 4 + c] = luma_dc[i];
                }
                inverse_hadamard_4x4(&mut luma_dc_raster);
                dequant_luma_dc_i16x16(&mut luma_dc_raster, qp_y, pps.scaling_list_4x4[0][0]);

                const DC_RASTER_TO_BLOCK: [usize; 16] = [
                    0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15,
                ];

                let mut luma_residual = [0i32; 256];
                for blk in 0..16 {
                    let (blk_row, blk_col) = BLOCK_INDEX_TO_OFFSET[blk];
                    let mut block_raster = [0i32; 16];
                    let dc_idx =
                        DC_RASTER_TO_BLOCK.iter().position(|&b| b == blk).unwrap();
                    block_raster[0] = luma_dc_raster[dc_idx];

                    if cbp_luma != 0 {
                        for scan_idx in 0..15 {
                            let (r, c) = ZIGZAG_4X4[scan_idx + 1];
                            block_raster[r * 4 + c] = luma_ac_scan[blk][scan_idx];
                        }
                        dequant_4x4_ac_raster(&mut block_raster, qp_y, &pps.scaling_list_4x4[0]);
                    }

                    inverse_dct_4x4(&mut block_raster);

                    for r in 0..4 {
                        for c in 0..4 {
                            luma_residual[(blk_row + r) * 16 + blk_col + c] =
                                block_raster[r * 4 + c];
                        }
                    }
                }

                // I16x16 prediction
                let mut luma_pred = [0u8; 256];
                let above: Option<Vec<u8>> = if mb_y > 0 {
                    Some(
                        (0..16)
                            .map(|x| frame.y[(mb_y - 1) * stride + mb_x + x])
                            .collect(),
                    )
                } else {
                    None
                };
                let left: Option<Vec<u8>> = if mb_x > 0 {
                    Some(
                        (0..16)
                            .map(|y| frame.y[(mb_y + y) * stride + mb_x - 1])
                            .collect(),
                    )
                } else {
                    None
                };
                let above_left = if mb_x > 0 && mb_y > 0 {
                    Some(frame.y[(mb_y - 1) * stride + mb_x - 1])
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

                for y in 0..16 {
                    for x in 0..16 {
                        let val = (luma_pred[y * 16 + x] as i32
                            + luma_residual[y * 16 + x])
                            .clamp(0, 255) as u8;
                        frame.y[(mb_y + y) * stride + mb_x + x] = val;
                    }
                }
            } else if mb_type == 25 {
                // === I_PCM macroblock ===
                reader.align_to_byte();
                for r in 0..16 {
                    for c in 0..16 {
                        frame.y[(mb_y + r) * stride + mb_x + c] =
                            reader.read_bits(8)? as u8;
                    }
                }
                let cw = (width / 2) as usize;
                let cx = mb_x / 2;
                let cy = mb_y / 2;
                for r in 0..8 {
                    for c in 0..8 {
                        frame.u[(cy + r) * cw + cx + c] = reader.read_bits(8)? as u8;
                    }
                }
                for r in 0..8 {
                    for c in 0..8 {
                        frame.v[(cy + r) * cw + cx + c] = reader.read_bits(8)? as u8;
                    }
                }
                for blk in 0..16 {
                    nc_luma[mb_idx * 16 + blk] = 16;
                }
                for blk in 0..4 {
                    nc_cb[mb_idx * 4 + blk] = 16;
                    nc_cr[mb_idx * 4 + blk] = 16;
                }
                mb_info[mb_idx] = MbInfo {
                    mb_type: MbType::Ipcm,
                    qp_y: 0,
                };
                prev_mb_qp = 0;
                continue;
            } else {
                return Err(DecodeError::from("unsupported mb_type for I slice"));
            }

            mb_info[mb_idx] = MbInfo {
                mb_type: MbType::Intra,
                qp_y,
            };

            // Intra MBs keep ref_idx=-1 (default) and mv=(0,0) (default).
            // The -1 ref_idx ensures predict_mv's match_count logic correctly
            // excludes intra neighbors from directional prediction.

            // === Reconstruct chroma (shared by I4x4 and I16x16) ===
            let mut chroma_dc_cb = [0i32; 4];
            let mut chroma_dc_cr = [0i32; 4];
            if cbp_chroma >= 1 {
                parse_residual_block_cavlc(&mut reader, &mut chroma_dc_cb, 4, -1)?;
                parse_residual_block_cavlc(&mut reader, &mut chroma_dc_cr, 4, -1)?;
            }

            let mut chroma_ac_scan_cb = [[0i32; 15]; 4];
            let mut chroma_ac_scan_cr = [[0i32; 15]; 4];
            if cbp_chroma >= 2 {
                for blk in 0..4 {
                    let nc = compute_nc(&nc_cb, mb_idx, mb_width as usize, blk, 4);
                    let tc = parse_residual_block_cavlc(
                        &mut reader,
                        &mut chroma_ac_scan_cb[blk],
                        15,
                        nc,
                    )?;
                    nc_cb[mb_idx * 4 + blk] = tc;
                }
                for blk in 0..4 {
                    let nc = compute_nc(&nc_cr, mb_idx, mb_width as usize, blk, 4);
                    let tc = parse_residual_block_cavlc(
                        &mut reader,
                        &mut chroma_ac_scan_cr[blk],
                        15,
                        nc,
                    )?;
                    nc_cr[mb_idx * 4 + blk] = tc;
                }
            }

            let chroma_width = (width / 2) as usize;
            let chroma_mb_x = mb_x / 2;
            let chroma_mb_y = mb_y / 2;

            // Scaling list indices: 1=Intra Cb, 2=Intra Cr
            for (plane_dc, plane_ac_scan, plane_buf, scale_idx) in [
                (&mut chroma_dc_cb, &chroma_ac_scan_cb, &mut frame.u, 1usize),
                (&mut chroma_dc_cr, &chroma_ac_scan_cr, &mut frame.v, 2usize),
            ] {
                let chroma_scale = &pps.scaling_list_4x4[scale_idx];
                if cbp_chroma >= 1 {
                    inverse_hadamard_2x2(plane_dc);
                    dequant_chroma_dc(plane_dc, qp_c, chroma_scale[0]);
                }

                let mut chroma_residual = [0i32; 64];
                for blk in 0..4 {
                    let blk_row = (blk / 2) * 4;
                    let blk_col = (blk % 2) * 4;
                    let mut block_raster = [0i32; 16];
                    block_raster[0] = plane_dc[blk];

                    if cbp_chroma >= 2 {
                        for scan_idx in 0..15 {
                            let (r, c) = ZIGZAG_4X4[scan_idx + 1];
                            block_raster[r * 4 + c] = plane_ac_scan[blk][scan_idx];
                        }
                        dequant_4x4_ac_raster(&mut block_raster, qp_c, chroma_scale);
                    }

                    inverse_dct_4x4(&mut block_raster);

                    for r in 0..4 {
                        for c in 0..4 {
                            chroma_residual[(blk_row + r) * 8 + blk_col + c] =
                                block_raster[r * 4 + c];
                        }
                    }
                }

                let mut chroma_pred = [0u8; 64];
                let above_c: Option<Vec<u8>> = if chroma_mb_y > 0 {
                    Some(
                        (0..8)
                            .map(|x| {
                                plane_buf
                                    [(chroma_mb_y - 1) * chroma_width + chroma_mb_x + x]
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
                                plane_buf
                                    [(chroma_mb_y + y) * chroma_width + chroma_mb_x - 1]
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

                for y in 0..8 {
                    for x in 0..8 {
                        let val = (chroma_pred[y * 8 + x] as i32
                            + chroma_residual[y * 8 + x])
                            .clamp(0, 255) as u8;
                        plane_buf[(chroma_mb_y + y) * chroma_width + chroma_mb_x + x] =
                            val;
                    }
                }
            }
            mb_idx += 1;
        }

        // Apply deblocking filter after all MBs are decoded
        deblock::filter_frame(
            &mut frame,
            &mb_info,
            mb_width as usize,
            mb_height as usize,
            &header,
            pps.chroma_qp_index_offset,
        );

        // Compute POC and insert into DPB
        let poc = self.dpb.compute_poc(sps, &header, nal.nal_unit_type, nal.nal_ref_idc);

        if nal.nal_unit_type == NalUnitType::SliceIdr {
            self.dpb.clear();
        }

        let reference = if nal.nal_ref_idc > 0 {
            ReferenceStatus::ShortTerm
        } else {
            ReferenceStatus::Unused
        };

        let pic = Rc::new(DecodedPicture {
            y: frame.y.clone(),
            u: frame.u.clone(),
            v: frame.v.clone(),
            width: frame.width,
            height: frame.height,
            frame_num: header.frame_num,
            pic_order_cnt: poc,
        });

        self.dpb.insert(pic, reference);

        Ok(Some(frame))
    }
}

/// Motion vector prediction for P_Skip macroblocks.
/// Uses the standard median predictor (same as P_L0_16x16 with ref_idx=0).
fn predict_mv_skip(
    mv_store: &[[i16; 2]],
    ref_idx_store: &[i8],
    mb_idx: usize,
    mb_width: usize,
) -> (i16, i16) {
    predict_mv(mv_store, ref_idx_store, mb_idx, mb_width, 0, 16, 16, 0)
}

/// Motion vector prediction using the median of neighbors A, B, C (spec 8.4.1.3).
/// `part_idx`: partition index (0 for first/only partition).
/// `part_w`, `part_h`: partition dimensions.
/// `ref_idx`: reference index for this partition.
#[allow(clippy::too_many_arguments)]
fn predict_mv(
    mv_store: &[[i16; 2]],
    ref_idx_store: &[i8],
    mb_idx: usize,
    mb_width: usize,
    part_idx: usize,
    part_w: usize,
    part_h: usize,
    ref_idx: i8,
) -> (i16, i16) {
    let py_off = if part_h == 8 && part_w == 16 { part_idx * 8 } else { 0 };
    let px_off = if part_w == 8 && part_h == 16 { part_idx * 8 } else { 0 };

    // A: left neighbor (4x4 block to the left of partition's top-left)
    let a = get_mv_neighbor_left(mv_store, ref_idx_store, mb_idx, mb_width, py_off, px_off);

    // B: above neighbor
    let b = get_mv_neighbor_above(mv_store, ref_idx_store, mb_idx, mb_width, py_off, px_off);

    // C: above-right neighbor (or D: above-left if C unavailable)
    let c = get_mv_neighbor_above_right(
        mv_store, ref_idx_store, mb_idx, mb_width, py_off, px_off, part_w,
    )
    .or_else(|| get_mv_neighbor_above_left(mv_store, ref_idx_store, mb_idx, mb_width, py_off, px_off));

    // Special cases for 16x8 and 8x16 (spec 8.4.1.3.1)
    if part_w == 16 && part_h == 8 {
        if part_idx == 0 {
            if let Some((mv, ri)) = b { if ri == ref_idx { return (mv[0], mv[1]); } }
        } else if let Some((mv, ri)) = a { if ri == ref_idx { return (mv[0], mv[1]); } }
    }
    if part_w == 8 && part_h == 16 {
        if part_idx == 0 {
            if let Some((mv, ri)) = a { if ri == ref_idx { return (mv[0], mv[1]); } }
        } else if let Some((mv, ri)) = c { if ri == ref_idx { return (mv[0], mv[1]); } }
    }

    // Count how many neighbors match the target ref_idx (spec 8.4.1.3.1)
    let ref_a = a.map(|(_, r)| r).unwrap_or(-1);
    let ref_b = b.map(|(_, r)| r).unwrap_or(-1);
    let ref_c = c.map(|(_, r)| r).unwrap_or(-1);
    let match_count =
        (ref_a == ref_idx) as u8 + (ref_b == ref_idx) as u8 + (ref_c == ref_idx) as u8;

    // When exactly one neighbor matches, use that neighbor's MV directly
    if match_count == 1 {
        if ref_a == ref_idx {
            if let Some((mv, _)) = a { return (mv[0], mv[1]); }
        }
        if ref_b == ref_idx {
            if let Some((mv, _)) = b { return (mv[0], mv[1]); }
        }
        if ref_c == ref_idx {
            if let Some((mv, _)) = c { return (mv[0], mv[1]); }
        }
    }

    // Otherwise: median predictor
    let mv_a = a.map(|(mv, _)| mv).unwrap_or([0, 0]);
    let mv_b = b.map(|(mv, _)| mv).unwrap_or([0, 0]);
    let mv_c = c.map(|(mv, _)| mv).unwrap_or([0, 0]);

    let mut xs = [mv_a[0], mv_b[0], mv_c[0]];
    let mut ys = [mv_a[1], mv_b[1], mv_c[1]];
    xs.sort();
    ys.sort();
    (xs[1], ys[1])
}

/// Get MV/ref of the left neighbor for a partition.
fn get_mv_neighbor_left(
    mv_store: &[[i16; 2]], ref_idx_store: &[i8],
    mb_idx: usize, mb_width: usize, py_off: usize, px_off: usize,
) -> Option<([i16; 2], i8)> {
    let mb_col = mb_idx % mb_width;
    if px_off > 0 {
        // Left is within this MB
        let lr = py_off / 4;
        let lc = (px_off - 4) / 4;
        let blk = BLOCK_INDEX_TO_OFFSET.iter()
            .position(|&(br, bc)| br / 4 == lr && bc / 4 == lc)?;
        Some((mv_store[mb_idx * 16 + blk], ref_idx_store[mb_idx * 16 + blk]))
    } else if mb_col > 0 {
        // Left is in the left MB (rightmost column)
        let left_mb = mb_idx - 1;
        let lr = py_off / 4;
        let lc = 3; // rightmost 4x4 column
        let blk = BLOCK_INDEX_TO_OFFSET.iter()
            .position(|&(br, bc)| br / 4 == lr && bc / 4 == lc)?;
        Some((mv_store[left_mb * 16 + blk], ref_idx_store[left_mb * 16 + blk]))
    } else {
        None
    }
}

/// Get MV/ref of the above neighbor for a partition.
fn get_mv_neighbor_above(
    mv_store: &[[i16; 2]], ref_idx_store: &[i8],
    mb_idx: usize, mb_width: usize, py_off: usize, px_off: usize,
) -> Option<([i16; 2], i8)> {
    let mb_row = mb_idx / mb_width;
    if py_off > 0 {
        // Above is within this MB
        let lr = (py_off - 4) / 4;
        let lc = px_off / 4;
        let blk = BLOCK_INDEX_TO_OFFSET.iter()
            .position(|&(br, bc)| br / 4 == lr && bc / 4 == lc)?;
        Some((mv_store[mb_idx * 16 + blk], ref_idx_store[mb_idx * 16 + blk]))
    } else if mb_row > 0 {
        let above_mb = mb_idx - mb_width;
        let lr = 3; // bottom row
        let lc = px_off / 4;
        let blk = BLOCK_INDEX_TO_OFFSET.iter()
            .position(|&(br, bc)| br / 4 == lr && bc / 4 == lc)?;
        Some((mv_store[above_mb * 16 + blk], ref_idx_store[above_mb * 16 + blk]))
    } else {
        None
    }
}

/// Get MV/ref of the above-right neighbor for a partition.
fn get_mv_neighbor_above_right(
    mv_store: &[[i16; 2]], ref_idx_store: &[i8],
    mb_idx: usize, mb_width: usize, py_off: usize, px_off: usize, part_w: usize,
) -> Option<([i16; 2], i8)> {
    let mb_col = mb_idx % mb_width;
    let mb_row = mb_idx / mb_width;
    let right_col = px_off + part_w;

    if py_off > 0 {
        // Above-right within this MB
        if right_col < 16 {
            let lr = (py_off - 4) / 4;
            let lc = right_col / 4;
            let blk = BLOCK_INDEX_TO_OFFSET.iter()
                .position(|&(br, bc)| br / 4 == lr && bc / 4 == lc)?;
            Some((mv_store[mb_idx * 16 + blk], ref_idx_store[mb_idx * 16 + blk]))
        } else {
            None // right edge of MB, above-right is in MB above-right
        }
    } else if mb_row > 0 {
        // Above-right in the MB above (or above-right MB)
        if right_col < 16 {
            let above_mb = mb_idx - mb_width;
            let lr = 3;
            let lc = right_col / 4;
            let blk = BLOCK_INDEX_TO_OFFSET.iter()
                .position(|&(br, bc)| br / 4 == lr && bc / 4 == lc)?;
            Some((mv_store[above_mb * 16 + blk], ref_idx_store[above_mb * 16 + blk]))
        } else if mb_col + 1 < mb_width {
            let above_right_mb = mb_idx - mb_width + 1;
            let blk = BLOCK_INDEX_TO_OFFSET.iter()
                .position(|&(br, bc)| br / 4 == 3 && bc / 4 == 0)?;
            Some((mv_store[above_right_mb * 16 + blk], ref_idx_store[above_right_mb * 16 + blk]))
        } else {
            None
        }
    } else {
        None
    }
}

/// Get MV/ref of the above-left neighbor for a partition (fallback for C).
fn get_mv_neighbor_above_left(
    mv_store: &[[i16; 2]], ref_idx_store: &[i8],
    mb_idx: usize, mb_width: usize, py_off: usize, px_off: usize,
) -> Option<([i16; 2], i8)> {
    let mb_col = mb_idx % mb_width;
    let mb_row = mb_idx / mb_width;

    if py_off > 0 && px_off > 0 {
        let lr = (py_off - 4) / 4;
        let lc = (px_off - 4) / 4;
        let blk = BLOCK_INDEX_TO_OFFSET.iter()
            .position(|&(br, bc)| br / 4 == lr && bc / 4 == lc)?;
        Some((mv_store[mb_idx * 16 + blk], ref_idx_store[mb_idx * 16 + blk]))
    } else if py_off == 0 && px_off == 0 && mb_row > 0 && mb_col > 0 {
        // Above-left MB, bottom-right block
        let al_mb = mb_idx - mb_width - 1;
        let blk = BLOCK_INDEX_TO_OFFSET.iter()
            .position(|&(br, bc)| br / 4 == 3 && bc / 4 == 3)?;
        Some((mv_store[al_mb * 16 + blk], ref_idx_store[al_mb * 16 + blk]))
    } else if py_off == 0 && px_off > 0 && mb_row > 0 {
        let above_mb = mb_idx - mb_width;
        let lc = (px_off - 4) / 4;
        let blk = BLOCK_INDEX_TO_OFFSET.iter()
            .position(|&(br, bc)| br / 4 == 3 && bc / 4 == lc)?;
        Some((mv_store[above_mb * 16 + blk], ref_idx_store[above_mb * 16 + blk]))
    } else if py_off > 0 && px_off == 0 && mb_col > 0 {
        let left_mb = mb_idx - 1;
        let lr = (py_off - 4) / 4;
        let blk = BLOCK_INDEX_TO_OFFSET.iter()
            .position(|&(br, bc)| br / 4 == lr && bc / 4 == 3)?;
        Some((mv_store[left_mb * 16 + blk], ref_idx_store[left_mb * 16 + blk]))
    } else {
        None
    }
}

/// Predict I4x4 mode from left (A) and above (B) neighbor block modes.
/// H.264 spec 8.3.1.1: predicted mode = min(modeA, modeB).
/// If either neighbor is unavailable, predicted mode is DC (2).
fn predict_i4x4_mode(
    modes: &[u8],
    mb_idx: usize,
    mb_width: usize,
    blk_idx: usize,
) -> u8 {
    // None = neighbor unavailable (picture boundary or non-I4x4 neighbor MB that
    // doesn't exist). When either is None, predicted mode defaults to DC (2).
    let mode_a = get_neighbor_i4x4_mode(modes, mb_idx, mb_width, blk_idx, true);
    let mode_b = get_neighbor_i4x4_mode(modes, mb_idx, mb_width, blk_idx, false);
    match (mode_a, mode_b) {
        (Some(a), Some(b)) => a.min(b),
        _ => 2, // DC when either neighbor is unavailable
    }
}

fn get_neighbor_i4x4_mode(
    modes: &[u8],
    mb_idx: usize,
    mb_width: usize,
    blk_idx: usize,
    is_left: bool,
) -> Option<u8> {
    // Block layout:  0  1 | 4  5
    //                2  3 | 6  7
    //               ------+------
    //                8  9 |12 13
    //               10 11 |14 15
    if is_left {
        match blk_idx {
            1 | 5 | 9 | 13 => Some(modes[mb_idx * 16 + blk_idx - 1]),
            3 | 7 | 11 | 15 => Some(modes[mb_idx * 16 + blk_idx - 1]),
            4 => Some(modes[mb_idx * 16 + 1]),
            6 => Some(modes[mb_idx * 16 + 3]),
            12 => Some(modes[mb_idx * 16 + 9]),
            14 => Some(modes[mb_idx * 16 + 11]),
            0 | 2 | 8 | 10 => {
                // Left edge of MB
                if !mb_idx.is_multiple_of(mb_width) {
                    let left_mb = mb_idx - 1;
                    let left_blk = match blk_idx {
                        0 => 5, 2 => 7, 8 => 13, 10 => 15, _ => unreachable!(),
                    };
                    Some(modes[left_mb * 16 + left_blk])
                } else {
                    None // picture left boundary
                }
            }
            _ => None,
        }
    } else {
        match blk_idx {
            2 | 6 | 10 | 14 => Some(modes[mb_idx * 16 + blk_idx - 2]),
            3 | 7 | 11 | 15 => Some(modes[mb_idx * 16 + blk_idx - 2]),
            8 => Some(modes[mb_idx * 16 + 2]),
            9 => Some(modes[mb_idx * 16 + 3]),
            12 => Some(modes[mb_idx * 16 + 6]),
            13 => Some(modes[mb_idx * 16 + 7]),
            0 | 1 | 4 | 5 => {
                // Top edge of MB
                if mb_idx >= mb_width {
                    let above_mb = mb_idx - mb_width;
                    let above_blk = match blk_idx {
                        0 => 10, 1 => 11, 4 => 14, 5 => 15, _ => unreachable!(),
                    };
                    Some(modes[above_mb * 16 + above_blk])
                } else {
                    None // picture top boundary
                }
            }
            _ => None,
        }
    }
}

/// Compute nC for a 4x4 block from left (A) and above (B) neighbors.
/// H.264 spec 9.2.1: nC = average of neighbor total_coeff values.
fn compute_nc(
    nc_array: &[u8],
    mb_idx: usize,
    mb_width: usize,
    blk_idx: usize,
    blks_per_mb: usize,
) -> i32 {
    let (left_blk, left_in_mb) = if blks_per_mb == 16 {
        match blk_idx {
            0 => (5usize, false),
            2 => (7, false),
            8 => (13, false),
            10 => (15, false),
            4 => (1, true),
            6 => (3, true),
            12 => (9, true),
            14 => (11, true),
            1 => (0, true),
            3 => (2, true),
            5 => (4, true),
            7 => (6, true),
            9 => (8, true),
            11 => (10, true),
            13 => (12, true),
            15 => (14, true),
            _ => unreachable!(),
        }
    } else {
        match blk_idx {
            0 => (1, false),
            2 => (3, false),
            1 => (0, true),
            3 => (2, true),
            _ => unreachable!(),
        }
    };

    let nc_a: Option<u8> = if left_in_mb {
        Some(nc_array[mb_idx * blks_per_mb + left_blk])
    } else if !mb_idx.is_multiple_of(mb_width) {
        Some(nc_array[(mb_idx - 1) * blks_per_mb + left_blk])
    } else {
        None
    };

    let (above_blk, above_in_mb) = if blks_per_mb == 16 {
        match blk_idx {
            0 => (10usize, false),
            1 => (11, false),
            4 => (14, false),
            5 => (15, false),
            2 => (0, true),
            3 => (1, true),
            6 => (4, true),
            7 => (5, true),
            8 => (2, true),
            9 => (3, true),
            10 => (8, true),
            11 => (9, true),
            12 => (6, true),
            13 => (7, true),
            14 => (12, true),
            15 => (13, true),
            _ => unreachable!(),
        }
    } else {
        match blk_idx {
            0 => (2, false),
            1 => (3, false),
            2 => (0, true),
            3 => (1, true),
            _ => unreachable!(),
        }
    };

    let nc_b: Option<u8> = if above_in_mb {
        Some(nc_array[mb_idx * blks_per_mb + above_blk])
    } else if mb_idx >= mb_width {
        Some(nc_array[(mb_idx - mb_width) * blks_per_mb + above_blk])
    } else {
        None
    };

    match (nc_a, nc_b) {
        (Some(a), Some(b)) => (a as i32 + b as i32 + 1) >> 1 ,
        (Some(n), None) | (None, Some(n)) => n as i32,
        (None, None) => 0,
    }
}

/// Dequantize AC coefficients in a 4x4 block in raster order (skip DC at [0][0]).
fn dequant_4x4_ac_raster(block: &mut [i32; 16], qp: i32, scale: &[u8; 16]) {
    use crate::residual::ZIGZAG_4X4;

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
                continue; // DC already handled
            }
            let idx = r * 4 + c;
            if block[idx] != 0 {
                let pc = match (r % 2, c % 2) {
                    (0, 0) => 0,
                    (1, 1) => 1,
                    _ => 2,
                };
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

    #[test]
    fn test_decode_multi_mb_frame() {
        let h264_data = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/multi_mb_frame.h264"
        ))
        .unwrap();
        let expected_yuv = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/multi_mb_frame.yuv"
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

        assert_eq!(frame.width, 64);
        assert_eq!(frame.height, 64);

        let mut output = Vec::new();
        output.extend_from_slice(&frame.y);
        output.extend_from_slice(&frame.u);
        output.extend_from_slice(&frame.v);
        assert_eq!(output, expected_yuv);
    }

    #[test]
    fn test_decode_i4x4_frame() {
        let h264_data = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/i4x4_frame.h264"
        ))
        .unwrap();
        let expected_yuv = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/i4x4_frame.yuv"
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
        assert_eq!(output, expected_yuv);
    }

    #[test]
    fn test_decode_deblock_frame() {
        let h264_data = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/deblock_frame.h264"
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

        assert_eq!(frame.width, 64);
        assert_eq!(frame.height, 64);

        // Write decoded output for reference generation
        let mut output = Vec::new();
        output.extend_from_slice(&frame.y);
        output.extend_from_slice(&frame.u);
        output.extend_from_slice(&frame.v);
        assert_eq!(output.len(), 6144);

        let expected_yuv = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/deblock_frame.yuv"
        ))
        .unwrap();
        assert_eq!(output, expected_yuv);
    }

    #[test]
    fn test_decode_mixed_i4x4_i16x16_frame() {
        let h264_data = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/mixed_i4x4_frame.h264"
        ))
        .unwrap();
        let expected_yuv = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/mixed_i4x4_frame.yuv"
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

        assert_eq!(frame.width, 64);
        assert_eq!(frame.height, 64);

        let mut output = Vec::new();
        output.extend_from_slice(&frame.y);
        output.extend_from_slice(&frame.u);
        output.extend_from_slice(&frame.v);
        assert_eq!(output, expected_yuv);
    }

    /// Helper to decode a test file and compare against reference YUV.
    fn decode_and_compare(h264_name: &str, expected_width: u32, expected_height: u32) {
        let h264_path = format!(
            "{}/testdata/{}.h264",
            env!("CARGO_MANIFEST_DIR"),
            h264_name
        );
        let yuv_path = format!(
            "{}/testdata/{}.yuv",
            env!("CARGO_MANIFEST_DIR"),
            h264_name
        );
        let h264_data = std::fs::read(&h264_path)
            .unwrap_or_else(|e| panic!("failed to read {}: {}", h264_path, e));
        let expected_yuv = std::fs::read(&yuv_path)
            .unwrap_or_else(|e| panic!("failed to read {}: {}", yuv_path, e));

        let nals = parse_annex_b(&h264_data);
        let mut decoder = Decoder::new();
        let mut frame = None;
        for nal in &nals {
            if let Some(f) = decoder.decode_nal(nal).unwrap() {
                frame = Some(f);
            }
        }
        let frame = frame.expect("should have decoded a frame");

        assert_eq!(frame.width, expected_width);
        assert_eq!(frame.height, expected_height);

        let mut output = Vec::new();
        output.extend_from_slice(&frame.y);
        output.extend_from_slice(&frame.u);
        output.extend_from_slice(&frame.v);
        assert_eq!(output, expected_yuv);
    }

    #[test]
    fn test_gradient_48x32() {
        // 3x2 MBs, QP=24, mixed I4x4/I16x16 (66.7% I4x4), gradient luma + colored chroma
        decode_and_compare("gradient_48x32", 48, 32);
    }

    #[test]
    fn test_edges_32x32_qp10() {
        // 2x2 MBs, QP=10 (high quality), high-contrast 8-pixel bar pattern
        decode_and_compare("edges_32x32_qp10", 32, 32);
    }

    #[test]
    fn test_edges_32x32_qp35() {
        // 2x2 MBs, QP=35 (low quality, heavy quantization)
        decode_and_compare("edges_32x32_qp35", 32, 32);
    }

    #[test]
    fn test_smooth_80x48() {
        // 5x3 MBs, QP=22, gentle luma gradient with non-trivial chroma
        decode_and_compare("smooth_80x48", 80, 48);
    }

    #[test]
    fn test_noise_16x16_qp12() {
        // Single MB, QP=12, pseudo-random content stressing CAVLC with many non-zero coefficients
        decode_and_compare("noise_16x16_qp12", 16, 16);
    }


    #[test]
    fn test_scaling_list() {
        decode_and_compare("scaling_test", 32, 32);
    }

    #[test]
    fn test_p_frame() {
        // 32x32, 2 frames: IDR + P-slice with motion (bars shifted right)
        let h264_data = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/p_frame_test.h264"
        ))
        .unwrap();
        let expected_yuv = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/p_frame_test.yuv"
        ))
        .unwrap();

        let nals = parse_annex_b(&h264_data);
        let mut decoder = Decoder::new();
        let mut frames = Vec::new();
        for nal in &nals {
            if let Some(f) = decoder.decode_nal(nal).unwrap() {
                frames.push(f);
            }
        }
        assert_eq!(frames.len(), 2, "should decode 2 frames (IDR + P)");
        assert_eq!(frames[0].width, 32);
        assert_eq!(frames[1].width, 32);

        // Compare both frames concatenated
        let mut output = Vec::new();
        for frame in &frames {
            output.extend_from_slice(&frame.y);
            output.extend_from_slice(&frame.u);
            output.extend_from_slice(&frame.v);
        }
        assert_eq!(output, expected_yuv);
    }

    /// Helper to decode a multi-frame test and compare all frames against reference.
    fn decode_multiframe_and_compare(
        h264_name: &str,
        expected_frames: usize,
        expected_width: u32,
        expected_height: u32,
    ) {
        let h264_path = format!(
            "{}/testdata/{}.h264",
            env!("CARGO_MANIFEST_DIR"),
            h264_name
        );
        let yuv_path = format!(
            "{}/testdata/{}.yuv",
            env!("CARGO_MANIFEST_DIR"),
            h264_name
        );
        let h264_data = std::fs::read(&h264_path)
            .unwrap_or_else(|e| panic!("failed to read {}: {}", h264_path, e));
        let expected_yuv = std::fs::read(&yuv_path)
            .unwrap_or_else(|e| panic!("failed to read {}: {}", yuv_path, e));

        let nals = parse_annex_b(&h264_data);
        let mut decoder = Decoder::new();
        let mut frames = Vec::new();
        for nal in &nals {
            if let Some(f) = decoder.decode_nal(nal).unwrap() {
                frames.push(f);
            }
        }
        assert_eq!(
            frames.len(),
            expected_frames,
            "expected {} frames",
            expected_frames
        );
        for f in &frames {
            assert_eq!(f.width, expected_width);
            assert_eq!(f.height, expected_height);
        }

        let mut output = Vec::new();
        for frame in &frames {
            output.extend_from_slice(&frame.y);
            output.extend_from_slice(&frame.u);
            output.extend_from_slice(&frame.v);
        }
        assert_eq!(output, expected_yuv);
    }

    #[test]
    fn test_p_multi_frame() {
        // 64x64, 4 frames: IDR + 3 P-frames with P16x16 (68.8%), P16x8/8x16 (14.6%),
        // I16x16-in-P (16.7%), moving diagonal gradient
        decode_multiframe_and_compare("p_multi_frame", 4, 64, 64);
    }

    #[test]
    fn test_p_skip_heavy() {
        // 64x32, 3 frames: IDR + 2 P with 50% skip, 37.5% I4x4-in-P, 12.5% P16x8/8x16,
        // mostly static with small moving region
        decode_multiframe_and_compare("p_skip_heavy", 3, 64, 32);
    }
}
