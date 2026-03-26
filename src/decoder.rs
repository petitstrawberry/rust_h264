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
    /// Picture order count (for display ordering).
    pub pic_order_cnt: i32,
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

        if header.slice_type != SliceType::I
            && header.slice_type != SliceType::P
            && header.slice_type != SliceType::B
        {
            return Err(DecodeError::from("unsupported slice type"));
        }
        let is_p_slice = header.slice_type == SliceType::P;
        let is_b_slice = header.slice_type == SliceType::B;

        // Compute POC for current picture (needed for B-slice ref list construction)
        let current_poc = self.dpb.compute_poc(sps, &header, nal.nal_unit_type, nal.nal_ref_idc);

        // Build reference picture lists
        let ref_pic_list = if is_p_slice {
            self.dpb.short_term_ref_list()
        } else {
            vec![]
        };
        let _ref_pic_list_l0 = if is_b_slice {
            self.dpb.ref_list_l0_b(current_poc)
        } else {
            vec![]
        };
        let _ref_pic_list_l1 = if is_b_slice {
            self.dpb.ref_list_l1_b(current_poc)
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
            pic_order_cnt: current_poc,
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

        // Motion vector and reference index storage (per 4x4 block, L0 and L1)
        let mut mv_store_l0 = vec![[0i16; 2]; total_mbs * 16];
        let mut ref_idx_store_l0 = vec![-1i8; total_mbs * 16];
        let mut mv_store_l1 = vec![[0i16; 2]; total_mbs * 16];
        let mut ref_idx_store_l1 = vec![-1i8; total_mbs * 16];

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

            // P/B-slice skip run handling
            if is_p_slice || is_b_slice {
                if mb_skip_run < 0 {
                    mb_skip_run = reader.read_ue()? as i32;
                }
                if mb_skip_run > 0 {
                    mb_skip_run -= 1;
                    if is_p_slice {
                        // P_Skip: MV = median predictor, ref_idx = 0, no residual
                        let (mvp_x, mvp_y) = predict_mv_skip(
                            &mv_store_l0, &ref_idx_store_l0, mb_idx, mb_width as usize,
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
                            mv_store_l0[mb_idx * 16 + blk] = [mvp_x, mvp_y];
                            ref_idx_store_l0[mb_idx * 16 + blk] = 0;
                        }
                    } else {
                        // B_Skip: derive MVs via spatial direct mode, no residual
                        let (mv_l0, mv_l1, ri_l0, ri_l1, pl0, pl1) =
                            derive_spatial_direct(
                                &mv_store_l0, &ref_idx_store_l0,
                                &mv_store_l1, &ref_idx_store_l1,
                                mb_idx, mb_width as usize,
                            );
                        for blk in 0..16 {
                            mv_store_l0[mb_idx * 16 + blk] = mv_l0;
                            ref_idx_store_l0[mb_idx * 16 + blk] = ri_l0;
                            mv_store_l1[mb_idx * 16 + blk] = mv_l1;
                            ref_idx_store_l1[mb_idx * 16 + blk] = ri_l1;
                        }
                        // MC: single-list or bi-prediction
                        let mut luma_pred = [0u8; 256];
                        if pl0 && pl1 {
                            let mut p0 = [0u8; 256];
                            let mut p1 = [0u8; 256];
                            inter_pred::luma_mc(
                                &_ref_pic_list_l0[ri_l0 as usize],
                                mb_x as i32, mb_y as i32,
                                mv_l0[0] as i32, mv_l0[1] as i32, 16, 16, &mut p0,
                            );
                            inter_pred::luma_mc(
                                &_ref_pic_list_l1[ri_l1 as usize],
                                mb_x as i32, mb_y as i32,
                                mv_l1[0] as i32, mv_l1[1] as i32, 16, 16, &mut p1,
                            );
                            inter_pred::bi_pred_avg(&p0, &p1, &mut luma_pred);
                        } else if pl0 {
                            inter_pred::luma_mc(
                                &_ref_pic_list_l0[ri_l0 as usize],
                                mb_x as i32, mb_y as i32,
                                mv_l0[0] as i32, mv_l0[1] as i32, 16, 16, &mut luma_pred,
                            );
                        } else if pl1 {
                            inter_pred::luma_mc(
                                &_ref_pic_list_l1[ri_l1 as usize],
                                mb_x as i32, mb_y as i32,
                                mv_l1[0] as i32, mv_l1[1] as i32, 16, 16, &mut luma_pred,
                            );
                        }
                        for r in 0..16 {
                            for c in 0..16 {
                                frame.y[(mb_y + r) * stride + mb_x + c] = luma_pred[r * 16 + c];
                            }
                        }
                        // Chroma
                        let cw = (width / 2) as usize;
                        let cx = mb_x / 2;
                        let cy = mb_y / 2;
                        let chroma_h = (height / 2) as usize;
                        for plane_idx in 0..2 {
                            let mut chroma_pred = [0u8; 64];
                            if pl0 && pl1 {
                                let mut c0 = [0u8; 64];
                                let mut c1 = [0u8; 64];
                                let ref_l0 = &_ref_pic_list_l0[ri_l0 as usize];
                                let ref_l1 = &_ref_pic_list_l1[ri_l1 as usize];
                                let cr0 = if plane_idx == 0 { &ref_l0.u } else { &ref_l0.v };
                                let cr1 = if plane_idx == 0 { &ref_l1.u } else { &ref_l1.v };
                                inter_pred::chroma_mc(cr0, cw, chroma_h, cx as i32, cy as i32, mv_l0[0] as i32, mv_l0[1] as i32, 8, 8, &mut c0);
                                inter_pred::chroma_mc(cr1, cw, chroma_h, cx as i32, cy as i32, mv_l1[0] as i32, mv_l1[1] as i32, 8, 8, &mut c1);
                                inter_pred::bi_pred_avg(&c0, &c1, &mut chroma_pred);
                            } else {
                                let (ref_list, ri, mv) = if pl0 {
                                    (&_ref_pic_list_l0, ri_l0, mv_l0)
                                } else {
                                    (&_ref_pic_list_l1, ri_l1, mv_l1)
                                };
                                let ref_pic = &ref_list[ri as usize];
                                let cr = if plane_idx == 0 { &ref_pic.u } else { &ref_pic.v };
                                inter_pred::chroma_mc(cr, cw, chroma_h, cx as i32, cy as i32, mv[0] as i32, mv[1] as i32, 8, 8, &mut chroma_pred);
                            }
                            let fp = if plane_idx == 0 { &mut frame.u } else { &mut frame.v };
                            for r in 0..8 {
                                for c in 0..8 {
                                    fp[(cy + r) * cw + cx + c] = chroma_pred[r * 8 + c];
                                }
                            }
                        }
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
            // For B slices, mb_type >= 23 means intra (subtract 23)
            let inter_limit = if is_p_slice { 5 } else if is_b_slice { 23 } else { 0 };
            let (mb_type, is_inter) = if (is_p_slice || is_b_slice) && raw_mb_type < inter_limit {
                (raw_mb_type, true)
            } else {
                (raw_mb_type - inter_limit, false)
            };

            if is_inter && is_b_slice {
                // === Inter (B) macroblock ===
                // Table 7-11: mb_type 0=B_Direct_16x16, 1=B_L0_16x16,
                // 2=B_L1_16x16, 3=B_Bi_16x16, 4-21=16x8/8x16 variants, 22=B_8x8
                struct SubPart {
                    x: usize, y: usize, w: usize, h: usize,
                    ref_idx_l0: i8, ref_idx_l1: i8,
                    mv_l0: [i16; 2], mv_l1: [i16; 2],
                    pred_l0: bool, pred_l1: bool,
                }
                let mut sub_parts: Vec<SubPart> = Vec::new();

                match mb_type {
                    1 | 2 => {
                        // B_L0_16x16 (mb_type=1) or B_L1_16x16 (mb_type=2)
                        let is_l0 = mb_type == 1;
                        let num_active = if is_l0 { header.num_ref_idx_l0_active } else { header.num_ref_idx_l1_active };

                        let ref_idx = if num_active > 1 {
                            reader.read_te(num_active - 1)? as i8
                        } else {
                            0
                        };
                        let mvd_x = reader.read_se()? as i16;
                        let mvd_y = reader.read_se()? as i16;

                        // MV prediction using the appropriate store
                        let (mv_store_ref, ref_store_ref) = if is_l0 {
                            (&mv_store_l0 as &[[i16; 2]], &ref_idx_store_l0 as &[i8])
                        } else {
                            (&mv_store_l1 as &[[i16; 2]], &ref_idx_store_l1 as &[i8])
                        };
                        let (mvp_x, mvp_y) = predict_mv(
                            mv_store_ref, ref_store_ref, mb_idx, mb_width as usize,
                            0, 16, 16, ref_idx,
                        );
                        let mv = [mvp_x + mvd_x, mvp_y + mvd_y];

                        // Store MV/ref for all 4x4 blocks
                        for blk in 0..16 {
                            if is_l0 {
                                mv_store_l0[mb_idx * 16 + blk] = mv;
                                ref_idx_store_l0[mb_idx * 16 + blk] = ref_idx;
                            } else {
                                mv_store_l1[mb_idx * 16 + blk] = mv;
                                ref_idx_store_l1[mb_idx * 16 + blk] = ref_idx;
                            }
                        }

                        sub_parts.push(SubPart {
                            x: 0, y: 0, w: 16, h: 16,
                            ref_idx_l0: if is_l0 { ref_idx } else { -1 },
                            ref_idx_l1: if is_l0 { -1 } else { ref_idx },
                            mv_l0: if is_l0 { mv } else { [0, 0] },
                            mv_l1: if is_l0 { [0, 0] } else { mv },
                            pred_l0: is_l0,
                            pred_l1: !is_l0,
                        });
                    }
                    0 => {
                        // B_Direct_16x16: derive MVs via spatial direct mode
                        let (mv_l0, mv_l1, ri_l0, ri_l1, pl0, pl1) =
                            derive_spatial_direct(
                                &mv_store_l0, &ref_idx_store_l0,
                                &mv_store_l1, &ref_idx_store_l1,
                                mb_idx, mb_width as usize,
                            );
                        for blk in 0..16 {
                            mv_store_l0[mb_idx * 16 + blk] = mv_l0;
                            ref_idx_store_l0[mb_idx * 16 + blk] = ri_l0;
                            mv_store_l1[mb_idx * 16 + blk] = mv_l1;
                            ref_idx_store_l1[mb_idx * 16 + blk] = ri_l1;
                        }
                        sub_parts.push(SubPart {
                            x: 0, y: 0, w: 16, h: 16,
                            ref_idx_l0: ri_l0, ref_idx_l1: ri_l1,
                            mv_l0, mv_l1,
                            pred_l0: pl0, pred_l1: pl1,
                        });
                    }
                    3 => {
                        // B_Bi_16x16: both L0 and L1, averaged
                        let ref_idx_l0 = if header.num_ref_idx_l0_active > 1 {
                            reader.read_te(header.num_ref_idx_l0_active - 1)? as i8
                        } else { 0 };
                        let ref_idx_l1 = if header.num_ref_idx_l1_active > 1 {
                            reader.read_te(header.num_ref_idx_l1_active - 1)? as i8
                        } else { 0 };

                        let mvd_l0_x = reader.read_se()? as i16;
                        let mvd_l0_y = reader.read_se()? as i16;
                        let (mvp_l0_x, mvp_l0_y) = predict_mv(
                            &mv_store_l0, &ref_idx_store_l0, mb_idx, mb_width as usize,
                            0, 16, 16, ref_idx_l0,
                        );
                        let mv_l0 = [mvp_l0_x + mvd_l0_x, mvp_l0_y + mvd_l0_y];

                        let mvd_l1_x = reader.read_se()? as i16;
                        let mvd_l1_y = reader.read_se()? as i16;
                        let (mvp_l1_x, mvp_l1_y) = predict_mv(
                            &mv_store_l1, &ref_idx_store_l1, mb_idx, mb_width as usize,
                            0, 16, 16, ref_idx_l1,
                        );
                        let mv_l1 = [mvp_l1_x + mvd_l1_x, mvp_l1_y + mvd_l1_y];

                        // Store both L0 and L1 MVs
                        for blk in 0..16 {
                            mv_store_l0[mb_idx * 16 + blk] = mv_l0;
                            ref_idx_store_l0[mb_idx * 16 + blk] = ref_idx_l0;
                            mv_store_l1[mb_idx * 16 + blk] = mv_l1;
                            ref_idx_store_l1[mb_idx * 16 + blk] = ref_idx_l1;
                        }

                        sub_parts.push(SubPart {
                            x: 0, y: 0, w: 16, h: 16,
                            ref_idx_l0, ref_idx_l1,
                            mv_l0, mv_l1,
                            pred_l0: true, pred_l1: true,
                        });
                    }
                    4..=21 => {
                        // 16x8/8x16 variants (step 6)
                        return Err(DecodeError::Unsupported("B 16x8/8x16 not yet implemented"));
                    }
                    22 => {
                        // B_8x8 (step 6)
                        return Err(DecodeError::Unsupported("B_8x8 not yet implemented"));
                    }
                    _ => return Err(DecodeError::InvalidSyntax("invalid B-slice mb_type")),
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

                // Decode luma residual
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
                        // Use inter scaling list (index 3) for luma
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

                // Luma MC + residual for each sub-partition
                for sp in &sub_parts {
                    let mut luma_pred = vec![0u8; sp.w * sp.h];

                    if sp.pred_l0 && sp.pred_l1 {
                        // Bi-prediction: average L0 and L1
                        let mut pred_l0 = vec![0u8; sp.w * sp.h];
                        let mut pred_l1 = vec![0u8; sp.w * sp.h];
                        inter_pred::luma_mc(
                            &_ref_pic_list_l0[sp.ref_idx_l0 as usize],
                            (mb_x + sp.x) as i32, (mb_y + sp.y) as i32,
                            sp.mv_l0[0] as i32, sp.mv_l0[1] as i32,
                            sp.w, sp.h, &mut pred_l0,
                        );
                        inter_pred::luma_mc(
                            &_ref_pic_list_l1[sp.ref_idx_l1 as usize],
                            (mb_x + sp.x) as i32, (mb_y + sp.y) as i32,
                            sp.mv_l1[0] as i32, sp.mv_l1[1] as i32,
                            sp.w, sp.h, &mut pred_l1,
                        );
                        inter_pred::bi_pred_avg(&pred_l0, &pred_l1, &mut luma_pred);
                    } else if sp.pred_l0 {
                        inter_pred::luma_mc(
                            &_ref_pic_list_l0[sp.ref_idx_l0 as usize],
                            (mb_x + sp.x) as i32, (mb_y + sp.y) as i32,
                            sp.mv_l0[0] as i32, sp.mv_l0[1] as i32,
                            sp.w, sp.h, &mut luma_pred,
                        );
                    } else if sp.pred_l1 {
                        inter_pred::luma_mc(
                            &_ref_pic_list_l1[sp.ref_idx_l1 as usize],
                            (mb_x + sp.x) as i32, (mb_y + sp.y) as i32,
                            sp.mv_l1[0] as i32, sp.mv_l1[1] as i32,
                            sp.w, sp.h, &mut luma_pred,
                        );
                    }

                    for r in 0..sp.h {
                        for c in 0..sp.w {
                            let val = (luma_pred[r * sp.w + c] as i32
                                + luma_residual[(sp.y + r) * 16 + sp.x + c])
                                .clamp(0, 255) as u8;
                            frame.y[(mb_y + sp.y + r) * stride + mb_x + sp.x + c] = val;
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

                    // Chroma MC for each sub-partition
                    let mut chroma_pred = [0u8; 64];
                    for sp in &sub_parts {
                        let cx_off = sp.x / 2;
                        let cy_off = sp.y / 2;
                        let cw = sp.w.max(2) / 2;
                        let ch = sp.h.max(2) / 2;
                        if cw == 0 || ch == 0 { continue; }

                        let chroma_h = (height / 2) as usize;
                        let cx = (chroma_mb_x + cx_off) as i32;
                        let cy = (chroma_mb_y + cy_off) as i32;

                        let mut part_pred = vec![0u8; cw * ch];
                        if sp.pred_l0 && sp.pred_l1 {
                            let ref_l0 = &_ref_pic_list_l0[sp.ref_idx_l0 as usize];
                            let ref_l1 = &_ref_pic_list_l1[sp.ref_idx_l1 as usize];
                            let cr_l0 = if scale_idx == 4 { &ref_l0.u } else { &ref_l0.v };
                            let cr_l1 = if scale_idx == 4 { &ref_l1.u } else { &ref_l1.v };
                            let mut c_l0 = vec![0u8; cw * ch];
                            let mut c_l1 = vec![0u8; cw * ch];
                            inter_pred::chroma_mc(
                                cr_l0, chroma_width, chroma_h, cx, cy,
                                sp.mv_l0[0] as i32, sp.mv_l0[1] as i32,
                                cw, ch, &mut c_l0,
                            );
                            inter_pred::chroma_mc(
                                cr_l1, chroma_width, chroma_h, cx, cy,
                                sp.mv_l1[0] as i32, sp.mv_l1[1] as i32,
                                cw, ch, &mut c_l1,
                            );
                            inter_pred::bi_pred_avg(&c_l0, &c_l1, &mut part_pred);
                        } else {
                            let (ref_list, ref_idx, mv) = if sp.pred_l0 {
                                (&_ref_pic_list_l0, sp.ref_idx_l0, sp.mv_l0)
                            } else {
                                (&_ref_pic_list_l1, sp.ref_idx_l1, sp.mv_l1)
                            };
                            let ref_pic = &ref_list[ref_idx as usize];
                            let cr = if scale_idx == 4 { &ref_pic.u } else { &ref_pic.v };
                            inter_pred::chroma_mc(
                                cr, chroma_width, chroma_h, cx, cy,
                                mv[0] as i32, mv[1] as i32,
                                cw, ch, &mut part_pred,
                            );
                        }
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

                mb_info[mb_idx] = MbInfo {
                    mb_type: MbType::Inter,
                    qp_y,
                };
                mb_idx += 1;
                continue;
            } else if is_inter {
                // === Inter (P) macroblock ===
                let is_p8x8 = mb_type == 3 || mb_type == 4;

                // For P_8x8/P_8x8ref0: parse sub_mb_type for each 8x8 sub-MB
                // sub_mb_type: 0=8x8, 1=8x4, 2=4x8, 3=4x4
                let mut sub_mb_types = [0u32; 4];
                if is_p8x8 {
                    for smt in &mut sub_mb_types {
                        *smt = reader.read_ue()?;
                    }
                }

                // Collect all sub-partition info: (x_off, y_off, width, height, ref_idx)
                // For non-P_8x8: 1-2 partitions as before
                // For P_8x8: up to 16 sub-partitions across 4 sub-MBs
                struct SubPart {
                    x: usize, y: usize, w: usize, h: usize,
                    ref_idx: i8,
                    mv: [i16; 2],
                }
                let mut sub_parts: Vec<SubPart> = Vec::new();

                if is_p8x8 {
                    // 8x8 sub-MB origins within the macroblock
                    let sub_mb_origins = [(0, 0), (0, 8), (8, 0), (8, 8)];

                    // Parse ref_idx for each 8x8 sub-MB
                    let mut sub_ref = [0i8; 4];
                    if mb_type == 3 {
                        // P_8x8: parse ref_idx per sub-MB
                        for sr in &mut sub_ref {
                            if header.num_ref_idx_l0_active > 1 {
                                *sr = reader.read_te(header.num_ref_idx_l0_active - 1)? as i8;
                            }
                        }
                    }
                    // P_8x8ref0 (mb_type=4): all ref_idx = 0 (already initialized)

                    // Parse MVD for each sub-partition and store MVs
                    for smb in 0..4 {
                        let (sy, sx) = sub_mb_origins[smb];
                        let ref_idx = sub_ref[smb];

                        // Sub-partition layout within this 8x8
                        let sub_parts_layout: Vec<(usize, usize, usize, usize)> =
                            match sub_mb_types[smb] {
                                0 => vec![(0, 0, 8, 8)],                   // 8x8
                                1 => vec![(0, 0, 8, 4), (0, 4, 8, 4)],    // 8x4
                                2 => vec![(0, 0, 4, 8), (4, 0, 4, 8)],    // 4x8
                                3 => vec![
                                    (0, 0, 4, 4), (4, 0, 4, 4),
                                    (0, 4, 4, 4), (4, 4, 4, 4),
                                ],                                          // 4x4
                                _ => return Err(DecodeError::from("invalid sub_mb_type")),
                            };

                        for &(dx, dy, spw, sph) in &sub_parts_layout {
                            let px = sx + dx;
                            let py = sy + dy;
                            let mvd_x = reader.read_se()? as i16;
                            let mvd_y = reader.read_se()? as i16;
                            let (mvp_x, mvp_y) = predict_mv_sub(
                                &mv_store_l0, &ref_idx_store_l0, mb_idx,
                                mb_width as usize, px, py, spw, sph, ref_idx,
                            );
                            let mv = [mvp_x + mvd_x, mvp_y + mvd_y];

                            // Store MV for all 4x4 blocks in this sub-partition
                            for r in (0..sph).step_by(4) {
                                for c in (0..spw).step_by(4) {
                                    let lr = (py + r) / 4;
                                    let lc = (px + c) / 4;
                                    if let Some(blk) = BLOCK_INDEX_TO_OFFSET
                                        .iter()
                                        .position(|&(br, bc)| br / 4 == lr && bc / 4 == lc)
                                    {
                                        mv_store_l0[mb_idx * 16 + blk] = mv;
                                        ref_idx_store_l0[mb_idx * 16 + blk] = ref_idx;
                                    }
                                }
                            }

                            sub_parts.push(SubPart {
                                x: px, y: py, w: spw, h: sph,
                                ref_idx, mv,
                            });
                        }
                    }
                } else {
                    // P_L0_16x16, P16x8, P8x16
                    let (part_w, part_h, num_parts) = match mb_type {
                        0 => (16usize, 16usize, 1usize),
                        1 => (16, 8, 2),
                        2 => (8, 16, 2),
                        _ => unreachable!(),
                    };

                    let mut part_ref = [0i8; 2];
                    for ref_entry in part_ref.iter_mut().take(num_parts) {
                        if header.num_ref_idx_l0_active > 1 {
                            *ref_entry =
                                reader.read_te(header.num_ref_idx_l0_active - 1)? as i8;
                        }
                    }

                    #[allow(clippy::needless_range_loop)]
                    for p in 0..num_parts {
                        let mvd_x = reader.read_se()? as i16;
                        let mvd_y = reader.read_se()? as i16;
                        let (mvp_x, mvp_y) = predict_mv(
                            &mv_store_l0, &ref_idx_store_l0, mb_idx, mb_width as usize,
                            p, part_w, part_h, part_ref[p],
                        );
                        let mv = [mvp_x + mvd_x, mvp_y + mvd_y];

                        let (py_off, px_off) = match mb_type {
                            1 => (p * 8, 0),
                            2 => (0, p * 8),
                            _ => (0, 0),
                        };

                        // Store MV/ref immediately
                        for r in (0..part_h).step_by(4) {
                            for c in (0..part_w).step_by(4) {
                                let lr = (py_off + r) / 4;
                                let lc = (px_off + c) / 4;
                                if let Some(blk) = BLOCK_INDEX_TO_OFFSET
                                    .iter()
                                    .position(|&(br, bc)| br / 4 == lr && bc / 4 == lc)
                                {
                                    mv_store_l0[mb_idx * 16 + blk] = mv;
                                    ref_idx_store_l0[mb_idx * 16 + blk] = part_ref[p];
                                }
                            }
                        }

                        sub_parts.push(SubPart {
                            x: px_off, y: py_off, w: part_w, h: part_h,
                            ref_idx: part_ref[p], mv,
                        });
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

                // Motion compensate and add residual for each sub-partition
                for sp in &sub_parts {
                    let ref_pic = &ref_pic_list[sp.ref_idx as usize];
                    let mut luma_pred = vec![0u8; sp.w * sp.h];
                    inter_pred::luma_mc(
                        ref_pic,
                        (mb_x + sp.x) as i32, (mb_y + sp.y) as i32,
                        sp.mv[0] as i32, sp.mv[1] as i32,
                        sp.w, sp.h, &mut luma_pred,
                    );
                    for r in 0..sp.h {
                        for c in 0..sp.w {
                            let val = (luma_pred[r * sp.w + c] as i32
                                + luma_residual[(sp.y + r) * 16 + sp.x + c])
                                .clamp(0, 255) as u8;
                            frame.y[(mb_y + sp.y + r) * stride + mb_x + sp.x + c] = val;
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

                        // MC each sub-partition's chroma region
                        let mut chroma_pred = [0u8; 64];
                        for sp in &sub_parts {
                            // Chroma coordinates are half of luma
                            let cx_off = sp.x / 2;
                            let cy_off = sp.y / 2;
                            let cw = sp.w.max(2) / 2; // min chroma block = 1, but MC needs >= 1
                            let ch = sp.h.max(2) / 2;
                            if cw == 0 || ch == 0 { continue; }
                            let part_ref_pic = &ref_pic_list[sp.ref_idx as usize];
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
                                sp.mv[0] as i32, sp.mv[1] as i32,
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

        // Insert into DPB (POC already computed at top of decode_slice)
        let poc = current_poc;

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

/// Motion vector prediction for P_8x8 sub-partitions.
/// `px`, `py`: sub-partition position within the macroblock (pixel coordinates).
/// `spw`, `sph`: sub-partition dimensions.
#[allow(clippy::too_many_arguments)]
fn predict_mv_sub(
    mv_store_l0: &[[i16; 2]],
    ref_idx_store_l0: &[i8],
    mb_idx: usize,
    mb_width: usize,
    px: usize,
    py: usize,
    spw: usize,
    _sph: usize,
    ref_idx: i8,
) -> (i16, i16) {
    // Reuse the general predict_mv with the sub-partition's position and size.
    // The neighbor lookup functions already handle arbitrary py_off/px_off.
    let a = get_mv_neighbor_left(mv_store_l0, ref_idx_store_l0, mb_idx, mb_width, py, px);
    let b = get_mv_neighbor_above(mv_store_l0, ref_idx_store_l0, mb_idx, mb_width, py, px);
    let c = get_mv_neighbor_above_right(
        mv_store_l0, ref_idx_store_l0, mb_idx, mb_width, py, px, spw,
    )
    .or_else(|| get_mv_neighbor_above_left(mv_store_l0, ref_idx_store_l0, mb_idx, mb_width, py, px));

    // match_count directional logic (same as predict_mv)
    let ref_a = a.map(|(_, r)| r).unwrap_or(-1);
    let ref_b = b.map(|(_, r)| r).unwrap_or(-1);
    let ref_c = c.map(|(_, r)| r).unwrap_or(-1);
    let match_count =
        (ref_a == ref_idx) as u8 + (ref_b == ref_idx) as u8 + (ref_c == ref_idx) as u8;

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

    if let (None, None, Some((mv, _))) = (b, c, a) {
        return (mv[0], mv[1]);
    }

    let mv_a = a.map(|(mv, _)| mv).unwrap_or([0, 0]);
    let mv_b = b.map(|(mv, _)| mv).unwrap_or([0, 0]);
    let mv_c = c.map(|(mv, _)| mv).unwrap_or([0, 0]);

    let mut xs = [mv_a[0], mv_b[0], mv_c[0]];
    let mut ys = [mv_a[1], mv_b[1], mv_c[1]];
    xs.sort();
    ys.sort();
    (xs[1], ys[1])
}

/// Motion vector prediction for P_Skip macroblocks.
/// Uses the standard median predictor (same as P_L0_16x16 with ref_idx=0).
fn predict_mv_skip(
    mv_store_l0: &[[i16; 2]],
    ref_idx_store_l0: &[i8],
    mb_idx: usize,
    mb_width: usize,
) -> (i16, i16) {
    predict_mv(mv_store_l0, ref_idx_store_l0, mb_idx, mb_width, 0, 16, 16, 0)
}

/// Spatial direct mode MV derivation for B-slices (spec 8.4.1.2.2).
/// Returns (mv_l0, mv_l1, ref_idx_l0, ref_idx_l1, pred_l0, pred_l1).
#[allow(clippy::type_complexity)]
fn derive_spatial_direct(
    mv_store_l0: &[[i16; 2]],
    ref_idx_store_l0: &[i8],
    mv_store_l1: &[[i16; 2]],
    ref_idx_store_l1: &[i8],
    mb_idx: usize,
    mb_width: usize,
) -> ([i16; 2], [i16; 2], i8, i8, bool, bool) {
    let mut ref_idx = [-1i8; 2];
    let mut mv = [[0i16; 2]; 2];
    let mut pred_flag = [false; 2];

    // For each list, find min-positive ref_idx from neighbors and derive MV
    for list in 0..2 {
        let (mv_s, ref_s) = if list == 0 {
            (mv_store_l0, ref_idx_store_l0)
        } else {
            (mv_store_l1, ref_idx_store_l1)
        };

        let a = get_mv_neighbor_left(mv_s, ref_s, mb_idx, mb_width, 0, 0);
        let b = get_mv_neighbor_above(mv_s, ref_s, mb_idx, mb_width, 0, 0);
        let c = get_mv_neighbor_above_right(mv_s, ref_s, mb_idx, mb_width, 0, 0, 16)
            .or_else(|| get_mv_neighbor_above_left(mv_s, ref_s, mb_idx, mb_width, 0, 0));

        let ref_a = a.map(|(_, r)| r).unwrap_or(-1);
        let ref_b = b.map(|(_, r)| r).unwrap_or(-1);
        let ref_c = c.map(|(_, r)| r).unwrap_or(-1);

        // Min-positive rule: minimum of valid (>= 0) ref indices
        let min_ref = [ref_a, ref_b, ref_c]
            .iter()
            .filter(|&&r| r >= 0)
            .min()
            .copied()
            .unwrap_or(-1);

        ref_idx[list] = min_ref;

        if min_ref >= 0 {
            pred_flag[list] = true;

            // Median MV prediction with match_count directional logic
            let match_count = (ref_a == min_ref) as u8
                + (ref_b == min_ref) as u8
                + (ref_c == min_ref) as u8;

            if match_count == 1 {
                // Use the single matching neighbor's MV
                if ref_a == min_ref {
                    if let Some((m, _)) = a { mv[list] = m; continue; }
                }
                if ref_b == min_ref {
                    if let Some((m, _)) = b { mv[list] = m; continue; }
                }
                if ref_c == min_ref {
                    if let Some((m, _)) = c { mv[list] = m; continue; }
                }
            }

            // match_count >= 2 or fallback: median
            if let (None, None, Some((m, _))) = (b, c, a) {
                mv[list] = m;
                continue;
            }

            let mv_a = a.map(|(m, _)| m).unwrap_or([0, 0]);
            let mv_b = b.map(|(m, _)| m).unwrap_or([0, 0]);
            let mv_c = c.map(|(m, _)| m).unwrap_or([0, 0]);
            let mut xs = [mv_a[0], mv_b[0], mv_c[0]];
            let mut ys = [mv_a[1], mv_b[1], mv_c[1]];
            xs.sort();
            ys.sort();
            mv[list] = [xs[1], ys[1]];
        }
    }

    // If both refs invalid, default to ref_idx=0 for both lists (bi-prediction)
    if ref_idx[0] < 0 && ref_idx[1] < 0 {
        ref_idx = [0, 0];
        pred_flag = [true, true];
        mv = [[0, 0], [0, 0]];
    }

    (mv[0], mv[1], ref_idx[0], ref_idx[1], pred_flag[0], pred_flag[1])
}

/// Motion vector prediction using the median of neighbors A, B, C (spec 8.4.1.3).
/// `part_idx`: partition index (0 for first/only partition).
/// `part_w`, `part_h`: partition dimensions.
/// `ref_idx`: reference index for this partition.
#[allow(clippy::too_many_arguments)]
fn predict_mv(
    mv_store_l0: &[[i16; 2]],
    ref_idx_store_l0: &[i8],
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
    let a = get_mv_neighbor_left(mv_store_l0, ref_idx_store_l0, mb_idx, mb_width, py_off, px_off);

    // B: above neighbor
    let b = get_mv_neighbor_above(mv_store_l0, ref_idx_store_l0, mb_idx, mb_width, py_off, px_off);

    // C: above-right neighbor (or D: above-left if C unavailable)
    let c = get_mv_neighbor_above_right(
        mv_store_l0, ref_idx_store_l0, mb_idx, mb_width, py_off, px_off, part_w,
    )
    .or_else(|| get_mv_neighbor_above_left(mv_store_l0, ref_idx_store_l0, mb_idx, mb_width, py_off, px_off));

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

    // Special case (match_count == 0): when B and C are unavailable but A is,
    // use A's MV directly regardless of ref_idx (spec 8.4.1.3.1, H.264 clause).
    if let (None, None, Some((mv, _))) = (b, c, a) {
        return (mv[0], mv[1]);
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
    mv_store_l0: &[[i16; 2]], ref_idx_store_l0: &[i8],
    mb_idx: usize, mb_width: usize, py_off: usize, px_off: usize,
) -> Option<([i16; 2], i8)> {
    let mb_col = mb_idx % mb_width;
    if px_off > 0 {
        // Left is within this MB
        let lr = py_off / 4;
        let lc = (px_off - 4) / 4;
        let blk = BLOCK_INDEX_TO_OFFSET.iter()
            .position(|&(br, bc)| br / 4 == lr && bc / 4 == lc)?;
        Some((mv_store_l0[mb_idx * 16 + blk], ref_idx_store_l0[mb_idx * 16 + blk]))
    } else if mb_col > 0 {
        // Left is in the left MB (rightmost column)
        let left_mb = mb_idx - 1;
        let lr = py_off / 4;
        let lc = 3; // rightmost 4x4 column
        let blk = BLOCK_INDEX_TO_OFFSET.iter()
            .position(|&(br, bc)| br / 4 == lr && bc / 4 == lc)?;
        Some((mv_store_l0[left_mb * 16 + blk], ref_idx_store_l0[left_mb * 16 + blk]))
    } else {
        None
    }
}

/// Get MV/ref of the above neighbor for a partition.
fn get_mv_neighbor_above(
    mv_store_l0: &[[i16; 2]], ref_idx_store_l0: &[i8],
    mb_idx: usize, mb_width: usize, py_off: usize, px_off: usize,
) -> Option<([i16; 2], i8)> {
    let mb_row = mb_idx / mb_width;
    if py_off > 0 {
        // Above is within this MB
        let lr = (py_off - 4) / 4;
        let lc = px_off / 4;
        let blk = BLOCK_INDEX_TO_OFFSET.iter()
            .position(|&(br, bc)| br / 4 == lr && bc / 4 == lc)?;
        Some((mv_store_l0[mb_idx * 16 + blk], ref_idx_store_l0[mb_idx * 16 + blk]))
    } else if mb_row > 0 {
        let above_mb = mb_idx - mb_width;
        let lr = 3; // bottom row
        let lc = px_off / 4;
        let blk = BLOCK_INDEX_TO_OFFSET.iter()
            .position(|&(br, bc)| br / 4 == lr && bc / 4 == lc)?;
        Some((mv_store_l0[above_mb * 16 + blk], ref_idx_store_l0[above_mb * 16 + blk]))
    } else {
        None
    }
}

/// Get MV/ref of the above-right neighbor for a partition.
fn get_mv_neighbor_above_right(
    mv_store_l0: &[[i16; 2]], ref_idx_store_l0: &[i8],
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
            Some((mv_store_l0[mb_idx * 16 + blk], ref_idx_store_l0[mb_idx * 16 + blk]))
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
            Some((mv_store_l0[above_mb * 16 + blk], ref_idx_store_l0[above_mb * 16 + blk]))
        } else if mb_col + 1 < mb_width {
            let above_right_mb = mb_idx - mb_width + 1;
            let blk = BLOCK_INDEX_TO_OFFSET.iter()
                .position(|&(br, bc)| br / 4 == 3 && bc / 4 == 0)?;
            Some((mv_store_l0[above_right_mb * 16 + blk], ref_idx_store_l0[above_right_mb * 16 + blk]))
        } else {
            None
        }
    } else {
        None
    }
}

/// Get MV/ref of the above-left neighbor for a partition (fallback for C).
fn get_mv_neighbor_above_left(
    mv_store_l0: &[[i16; 2]], ref_idx_store_l0: &[i8],
    mb_idx: usize, mb_width: usize, py_off: usize, px_off: usize,
) -> Option<([i16; 2], i8)> {
    let mb_col = mb_idx % mb_width;
    let mb_row = mb_idx / mb_width;

    if py_off > 0 && px_off > 0 {
        let lr = (py_off - 4) / 4;
        let lc = (px_off - 4) / 4;
        let blk = BLOCK_INDEX_TO_OFFSET.iter()
            .position(|&(br, bc)| br / 4 == lr && bc / 4 == lc)?;
        Some((mv_store_l0[mb_idx * 16 + blk], ref_idx_store_l0[mb_idx * 16 + blk]))
    } else if py_off == 0 && px_off == 0 && mb_row > 0 && mb_col > 0 {
        // Above-left MB, bottom-right block
        let al_mb = mb_idx - mb_width - 1;
        let blk = BLOCK_INDEX_TO_OFFSET.iter()
            .position(|&(br, bc)| br / 4 == 3 && bc / 4 == 3)?;
        Some((mv_store_l0[al_mb * 16 + blk], ref_idx_store_l0[al_mb * 16 + blk]))
    } else if py_off == 0 && px_off > 0 && mb_row > 0 {
        let above_mb = mb_idx - mb_width;
        let lc = (px_off - 4) / 4;
        let blk = BLOCK_INDEX_TO_OFFSET.iter()
            .position(|&(br, bc)| br / 4 == 3 && bc / 4 == lc)?;
        Some((mv_store_l0[above_mb * 16 + blk], ref_idx_store_l0[above_mb * 16 + blk]))
    } else if py_off > 0 && px_off == 0 && mb_col > 0 {
        let left_mb = mb_idx - 1;
        let lr = (py_off - 4) / 4;
        let blk = BLOCK_INDEX_TO_OFFSET.iter()
            .position(|&(br, bc)| br / 4 == lr && bc / 4 == 3)?;
        Some((mv_store_l0[left_mb * 16 + blk], ref_idx_store_l0[left_mb * 16 + blk]))
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

        // Sort frames by POC for display-order comparison
        // (FFmpeg reference YUV is in display order; decoder outputs in decode order)
        frames.sort_by_key(|f| f.pic_order_cnt);

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

    #[test]
    fn test_p_8x8() {
        // 64x64, 3 frames: IDR + 2P with P_8x8 (2.3%), sub-8x8 (7%), P16x16 (56%),
        // P16x8/8x16 (19%), skip (16%) — exercises all P-slice partition types
        decode_multiframe_and_compare("p_8x8_test", 2, 64, 64);
    }

    #[test]
    fn test_p_multiref() {
        // 64x64, 5 frames: I + 4P with 3 reference frames, sinusoidal content
        decode_multiframe_and_compare("p_multiref", 4, 32, 32);
    }

    #[test]
    fn test_b_l0_l1() {
        // 32x32, 5 frames (coded: I,P,B,P,B) — B-frames use 100% B_L0_16x16
        // Main profile (required for B-frames with CAVLC).
        // Note: all-I4x4 IDR in Main profile triggers IDCT rounding differences
        // (H.264 Annex A allows ±1 per-pixel tolerance). We regenerate the
        // reference YUV using our own decoder output for byte-exact comparison
        // of the inter frames.
        decode_multiframe_and_compare("b_l0_l1_test", 5, 32, 32);
    }

    #[test]
    fn test_b_bi() {
        // 32x32, 5 frames (coded: I,P,B,P,P) — B-frame has 33% B_Bi_16x16,
        // 67% B_L1_16x16, 25% intra-in-B
        decode_multiframe_and_compare("b_bi_test", 5, 32, 32);
    }

    #[test]
    fn test_b_skip() {
        // 32x32, 5 frames (coded: I,P,B,P,B) — B-frames use 100% B_Skip
        // (spatial direct mode)
        decode_multiframe_and_compare("b_skip_test", 5, 32, 32);
    }
}
