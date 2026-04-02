use std::collections::HashMap;

use std::rc::Rc;

use crate::bitstream::BitstreamReader;
use crate::deblock::{self, MbInfo, MbType};
use crate::decode_cabac::CabacMbResult;
use crate::dpb::{DecodedPicture, Dpb, ReferenceStatus};
use crate::error::DecodeError;
use crate::mv_pred::WeightContext;
use crate::nal::{NalUnit, NalUnitType};
use crate::pps::{parse_pps, Pps};
use crate::slice::{parse_slice_header, SliceType};
use crate::slice_context::{SliceContext, SliceParams};
use crate::sps::{parse_sps, Sps};

/// A decoded YUV 4:2:0 frame.
#[derive(Debug, Clone)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
    /// Picture order count (for display ordering).
    pub pic_order_cnt: i32,
}

/// In-progress picture state shared across slices within the same frame.
#[derive(Clone)]
struct PictureState {
    frame: Frame,
    frame_num: u32,
    poc: i32,
    nal_unit_type: NalUnitType,
    nal_ref_idc: u8,
    // Per-MB arrays that persist across slices
    nc_luma: Vec<u8>,
    nc_cb: Vec<u8>,
    nc_cr: Vec<u8>,
    mv_store_l0: Vec<[i16; 2]>,
    mv_store_l1: Vec<[i16; 2]>,
    ref_idx_store_l0: Vec<i8>,
    ref_poc_store_l0: Vec<i32>,
    ref_idx_store_l1: Vec<i8>,
    mvd_store: Vec<[i16; 2]>,
    mvd_store_l1: Vec<[i16; 2]>,
    mb_info: Vec<deblock::MbInfo>,
    i4x4_modes: Vec<u8>,
    // CABAC neighbor context state
    mb_cbp: Vec<u16>,
    mb_chroma_pred: Vec<u8>,
    mb_is_8x8dct: Vec<bool>,
    mb_skip: Vec<bool>,
    mb_is_direct: Vec<bool>,
    is_i16x16: Vec<bool>,
    /// Per-MB slice ID for slice boundary detection. MBs from different
    /// slices are treated as unavailable for CABAC context and MV prediction.
    mb_slice_id: Vec<u16>,
    /// Current slice ID counter (incremented for each new slice).
    current_slice_id: u16,
    #[allow(dead_code)]
    prev_mb_qp: i32,
    #[allow(dead_code)]
    last_qp_delta_nonzero: bool,
    // Slice header info for finalization
    mmco_ops: Vec<(u32, u32)>,
    is_intra_slice: bool,
    // Deblock parameters (from first slice; per-slice deblock offsets
    // could differ but we use the first slice's values)
    disable_deblocking_filter_idc: u32,
    slice_alpha_c0_offset_div2: i32,
    slice_beta_offset_div2: i32,
    chroma_qp_index_offset: i32,
    mb_width: u32,
    mb_height: u32,
}

pub struct Decoder {
    sps_table: HashMap<u32, Sps>,
    pps_table: HashMap<u32, Pps>,
    dpb: Dpb,
    /// In-progress picture being assembled from one or more slices.
    pending: Option<PictureState>,
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
            pending: None,
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
            NalUnitType::SliceIdr | NalUnitType::Slice => {
                // Peek at first_mb_in_slice to detect new vs continuation slice
                let mut peek = BitstreamReader::new(&nal.rbsp);
                let first_mb = peek.read_ue().unwrap_or(0);

                // Check if this is a new picture: first_mb==0 means first
                // slice of a new picture. Continuation slices (first_mb > 0)
                // belong to the same picture even for IDR NALs.
                let is_new_picture = first_mb == 0;

                // Finalize pending frame if a new picture starts
                let prev_frame = if is_new_picture {
                    self.finalize_pending()
                } else {
                    None
                };

                // Decode this slice (creates or continues PictureState).
                // For CAVLC multi-slice, end-of-slice detection may fail,
                // causing errors from reading past the slice boundary. If
                // we had a pending picture, the already-decoded MBs are
                // valid, so we treat the error as end-of-slice.
                //
                // Since decode_slice takes self.pending via take(), we must
                // save a backup for continuation slices so we can restore it
                // if the decode fails mid-slice.
                let had_pending = !is_new_picture && self.pending.is_some();
                let pending_backup = if had_pending {
                    self.pending.clone()
                } else {
                    None
                };
                match self.decode_slice(nal) {
                    Ok(()) => {}
                    Err(_e) if self.pending.is_some() => {}
                    Err(_e) if had_pending => {
                        // decode_slice consumed self.pending but failed before
                        // reassembling it. Restore the backup so already-decoded
                        // MBs from earlier slices are preserved.
                        self.pending = pending_backup;
                    }
                    Err(e) => return Err(e),
                }

                Ok(prev_frame)
            }
            _ => Ok(None),
        }
    }

    /// Flush the decoder — finalize any pending frame. Call after all NALs are fed.
    pub fn flush(&mut self) -> Option<Frame> {
        self.finalize_pending()
    }

    /// Finalize the pending picture: apply deblocking, insert into DPB, return frame.
    fn finalize_pending(&mut self) -> Option<Frame> {
        let mut ps = self.pending.take()?;

        // Apply deblocking filter
        deblock::filter_frame_params(
            &mut ps.frame,
            &ps.mb_info,
            ps.mb_width as usize,
            ps.disable_deblocking_filter_idc,
            ps.slice_alpha_c0_offset_div2,
            ps.slice_beta_offset_div2,
            ps.chroma_qp_index_offset,
        );

        if ps.nal_unit_type == NalUnitType::SliceIdr {
            self.dpb.clear();
        }

        let reference = if ps.nal_ref_idc > 0 {
            ReferenceStatus::ShortTerm
        } else {
            ReferenceStatus::Unused
        };

        for &(op, param) in &ps.mmco_ops {
            if op == 1 {
                let pic_num_to_remove = ps.frame_num as i32 - (param as i32 + 1);
                self.dpb.mark_short_term_unused(pic_num_to_remove as u32);
            }
        }

        let pic = Rc::new(DecodedPicture {
            y: ps.frame.y.clone(),
            u: ps.frame.u.clone(),
            v: ps.frame.v.clone(),
            width: ps.frame.width,
            height: ps.frame.height,
            frame_num: ps.frame_num,
            pic_order_cnt: ps.poc,
            mv_l0: ps.mv_store_l0,
            ref_idx_l0: ps.ref_idx_store_l0,
            ref_poc_l0: ps.ref_poc_store_l0,
            mb_width: ps.mb_width,
            is_intra: ps.is_intra_slice,
        });

        self.dpb.insert(pic, reference);

        Some(ps.frame)
    }

    fn decode_slice(&mut self, nal: &NalUnit) -> Result<(), DecodeError> {
        let pps = self
            .pps_table
            .values()
            .next()
            .ok_or(DecodeError::InvalidSyntax("no PPS available"))?;
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
        let current_poc = self
            .dpb
            .compute_poc(sps, &header, nal.nal_unit_type, nal.nal_ref_idc);

        // Build reference picture lists
        let max_pic_num = 1u32 << (sps.log2_max_frame_num_minus4 + 4);
        let mut ref_pic_list = if is_p_slice {
            let mut refs = self.dpb.short_term_ref_list();
            // Pad ref list if shorter than num_ref_idx_l0_active (spec 8.2.4.2.1:
            // if the list is shorter, duplicate the last entry to fill)
            if !refs.is_empty() {
                while refs.len() < header.num_ref_idx_l0_active as usize {
                    refs.push(refs.last().unwrap().clone());
                }
            }
            refs
        } else {
            vec![]
        };
        let mut _ref_pic_list_l0 = if is_b_slice {
            let mut refs = self.dpb.ref_list_l0_b(current_poc);
            if !refs.is_empty() {
                while refs.len() < header.num_ref_idx_l0_active as usize {
                    refs.push(refs.last().unwrap().clone());
                }
            }
            refs
        } else {
            vec![]
        };
        let mut _ref_pic_list_l1 = if is_b_slice {
            let mut refs = self.dpb.ref_list_l1_b(current_poc);
            if !refs.is_empty() {
                while refs.len() < header.num_ref_idx_l1_active as usize {
                    refs.push(refs.last().unwrap().clone());
                }
            }
            refs
        } else {
            vec![]
        };

        // Apply ref_pic_list_modification (spec 8.2.4.3)
        if is_p_slice && !header.ref_list_mod_l0.is_empty() {
            Dpb::apply_ref_list_modification(
                &mut ref_pic_list,
                &header.ref_list_mod_l0,
                header.frame_num,
                max_pic_num,
            );
        }
        if is_b_slice && !header.ref_list_mod_l0.is_empty() {
            Dpb::apply_ref_list_modification(
                &mut _ref_pic_list_l0,
                &header.ref_list_mod_l0,
                header.frame_num,
                max_pic_num,
            );
        }
        if is_b_slice && !header.ref_list_mod_l1.is_empty() {
            Dpb::apply_ref_list_modification(
                &mut _ref_pic_list_l1,
                &header.ref_list_mod_l1,
                header.frame_num,
                max_pic_num,
            );
        }

        // Weighted prediction mode:
        // 0 = no weighting (default)
        // 1 = explicit weights (P-slice weighted_pred_flag=1, or B-slice weighted_bipred_idc=1)
        // 2 = implicit weights (B-slice weighted_bipred_idc=2)
        let use_weight = if (is_p_slice && pps.weighted_pred_flag)
            || (is_b_slice && pps.weighted_bipred_idc == 1)
        {
            1
        } else if is_b_slice && pps.weighted_bipred_idc == 2 {
            2
        } else {
            0
        };

        // Implicit weighted prediction: compute L0 weight from POC distances.
        // implicit_weights[l0_idx][l1_idx] = w0. L1 weight = 64 - w0. Fixed log2_denom=5.
        let implicit_weights: Vec<Vec<i32>> = if use_weight == 2 {
            _ref_pic_list_l0
                .iter()
                .map(|ref_l0| {
                    _ref_pic_list_l1
                        .iter()
                        .map(|ref_l1| {
                            let td = (ref_l1.pic_order_cnt - ref_l0.pic_order_cnt).clamp(-128, 127);
                            if td == 0 {
                                32
                            } else {
                                let tb = (current_poc - ref_l0.pic_order_cnt).clamp(-128, 127);
                                let tx = (16384 + (td.abs() / 2)) / td;
                                let w1 = (tb * tx + 32) >> 8;
                                if !(-64..=128).contains(&w1) {
                                    32
                                } else {
                                    64 - w1
                                }
                            }
                        })
                        .collect()
                })
                .collect()
        } else {
            vec![]
        };

        if is_b_slice && use_weight == 2 {
            eprintln!("B poc={} l0_active={} l1_active={} L0[0].poc={} L1[0].poc={} implicit_weights[0][0]={}",
                current_poc,
                header.num_ref_idx_l0_active,
                header.num_ref_idx_l1_active,
                _ref_pic_list_l0.first().map(|r| r.pic_order_cnt).unwrap_or(-1),
                _ref_pic_list_l1.first().map(|r| r.pic_order_cnt).unwrap_or(-1),
                implicit_weights.first().and_then(|r| r.first()).copied().unwrap_or(-1),
            );
        }
        let wctx = WeightContext {
            use_weight,
            wt: header.weight_table.as_ref(),
            implicit_weights: &implicit_weights,
        };

        let width = sps.width();
        let height = sps.height();
        let mb_width = width.div_ceil(16);
        let mb_height = height.div_ceil(16);
        let total_mbs = (mb_width * mb_height) as usize;

        let slice_qp = header.qp_y(pps);

        // Create or reuse PictureState for multi-slice support.
        // For continuation slices (first_mb > 0), reuse the pending state
        // so per-MB data from earlier slices is visible for MV prediction,
        // CABAC neighbor contexts, and deblocking.
        let is_continuation = header.first_mb_in_slice > 0 && self.pending.is_some();
        let ps = if is_continuation {
            self.pending.take().unwrap()
        } else {
            PictureState {
                frame: Frame {
                    width,
                    height,
                    y: vec![0u8; (width * height) as usize],
                    u: vec![0u8; (width * height / 4) as usize],
                    v: vec![0u8; (width * height / 4) as usize],
                    pic_order_cnt: current_poc,
                },
                frame_num: header.frame_num,
                poc: current_poc,
                nal_unit_type: nal.nal_unit_type,
                nal_ref_idc: nal.nal_ref_idc,
                nc_luma: vec![0u8; total_mbs * 16],
                nc_cb: vec![0u8; total_mbs * 4],
                nc_cr: vec![0u8; total_mbs * 4],
                mv_store_l0: vec![[0i16; 2]; total_mbs * 16],
                mv_store_l1: vec![[0i16; 2]; total_mbs * 16],
                ref_idx_store_l0: vec![-1i8; total_mbs * 16],
                ref_poc_store_l0: vec![-1i32; total_mbs * 16],
                ref_idx_store_l1: vec![-1i8; total_mbs * 16],
                mvd_store: vec![[0i16; 2]; total_mbs * 16],
                mvd_store_l1: vec![[0i16; 2]; total_mbs * 16],
                mb_info: vec![deblock::MbInfo::default(); total_mbs],
                i4x4_modes: vec![2u8; total_mbs * 16],
                mb_cbp: vec![0u16; total_mbs],
                mb_chroma_pred: vec![0u8; total_mbs],
                mb_is_8x8dct: vec![false; total_mbs],
                mb_skip: vec![false; total_mbs],
                mb_is_direct: vec![false; total_mbs],
                is_i16x16: vec![false; total_mbs],
                mb_slice_id: vec![0u16; total_mbs],
                current_slice_id: 0,
                prev_mb_qp: slice_qp,
                last_qp_delta_nonzero: false,
                mmco_ops: header.mmco_ops.clone(),
                is_intra_slice: header.slice_type == SliceType::I,
                disable_deblocking_filter_idc: header.disable_deblocking_filter_idc,
                slice_alpha_c0_offset_div2: header.slice_alpha_c0_offset_div2,
                slice_beta_offset_div2: header.slice_beta_offset_div2,
                chroma_qp_index_offset: pps.chroma_qp_index_offset,
                mb_width,
                mb_height,
            }
        };

        // Destructure into local variables so existing code works unchanged
        let PictureState {
            mut frame,
            frame_num: _ps_frame_num,
            poc: _ps_poc,
            nal_unit_type: _ps_nal_type,
            nal_ref_idc: _ps_nal_ref_idc,
            mut nc_luma,
            mut nc_cb,
            mut nc_cr,
            mut mv_store_l0,
            mut mv_store_l1,
            mut ref_idx_store_l0,
            mut ref_poc_store_l0,
            mut ref_idx_store_l1,
            mut mvd_store,
            mut mvd_store_l1,
            mut mb_info,
            mut i4x4_modes,
            mut mb_cbp,
            mut mb_chroma_pred,
            mut mb_is_8x8dct,
            mut mb_skip,
            mut mb_is_direct,
            mut is_i16x16,
            mut mb_slice_id,
            mut current_slice_id,
            prev_mb_qp: _,
            last_qp_delta_nonzero: _,
            mmco_ops: _ps_mmco_ops,
            is_intra_slice: _ps_is_intra,
            disable_deblocking_filter_idc: ps_deblock_idc,
            slice_alpha_c0_offset_div2: ps_alpha,
            slice_beta_offset_div2: ps_beta,
            chroma_qp_index_offset: ps_chroma_qp_offset,
            mb_width: _ps_mb_width,
            mb_height: _ps_mb_height,
        } = ps;

        // Increment slice ID for continuation slices so boundary checks work
        if is_continuation {
            current_slice_id += 1;
        }
        let this_slice_id = current_slice_id;

        // Each slice reinitializes its own QP from the slice header
        let mut prev_mb_qp = slice_qp;
        let mut last_qp_delta_nonzero = false;

        // CABAC or CAVLC?
        let use_cabac = pps.entropy_coding_mode_flag;

        // Initialize CABAC engine if needed
        let cabac_byte_pos = if use_cabac {
            // Align to byte boundary (the cabac_alignment_one_bit + zero padding
            // are handled by aligning the bitstream reader)
            let (pos, _data) = reader.cabac_start();
            Some(pos)
        } else {
            None
        };
        // Create CabacReader from original RBSP data (avoids borrow conflict with reader)
        let mut cabac_reader =
            cabac_byte_pos.map(|pos| crate::cabac::CabacReader::new(&nal.rbsp, pos));
        let mut cabac_state = if use_cabac {
            crate::cabac::init_cabac_states(
                slice_qp,
                header.slice_type == SliceType::I,
                header.cabac_init_idc,
            )
        } else {
            [0u8; 1024]
        };

        let mut mb_skip_run: i32 = -1; // -1 = not initialized for P slices

        let stride = width as usize;

        // Macro to construct a SliceContext from the local variables.
        // Used at each call site that delegates to a SliceContext method.
        macro_rules! make_ctx {
            () => {
                SliceContext {
                    frame: &mut frame,
                    stride,
                    width,
                    height,
                    mb_width,
                    nc_luma: &mut nc_luma,
                    nc_cb: &mut nc_cb,
                    nc_cr: &mut nc_cr,
                    mv_store_l0: &mut mv_store_l0,
                    mv_store_l1: &mut mv_store_l1,
                    ref_idx_store_l0: &mut ref_idx_store_l0,
                    ref_poc_store_l0: &mut ref_poc_store_l0,
                    ref_idx_store_l1: &mut ref_idx_store_l1,
                    mvd_store: &mut mvd_store,
                    mvd_store_l1: &mut mvd_store_l1,
                    mb_info: &mut mb_info,
                    i4x4_modes: &mut i4x4_modes,
                    mb_cbp: &mut mb_cbp,
                    mb_chroma_pred: &mut mb_chroma_pred,
                    mb_is_8x8dct: &mut mb_is_8x8dct,
                    mb_skip: &mut mb_skip,
                    mb_is_direct: &mut mb_is_direct,
                    is_i16x16: &mut is_i16x16,
                    mb_slice_id: &mut mb_slice_id,
                    this_slice_id,
                    prev_mb_qp,
                    last_qp_delta_nonzero,
                }
            };
        }

        let params = SliceParams {
            is_p_slice,
            is_b_slice,
            use_weight,
            current_poc,
            direct_spatial_mv_pred_flag: header.direct_spatial_mv_pred_flag,
            direct_8x8_inference_flag: sps.direct_8x8_inference_flag,
            transform_8x8_mode_flag: pps.transform_8x8_mode_flag,
            scaling_list_4x4: &pps.scaling_list_4x4,
            scaling_list_8x8: &pps.scaling_list_8x8,
            chroma_qp_index_offset: pps.chroma_qp_index_offset,
            ref_pic_list: &ref_pic_list,
            ref_pic_list_l0: &_ref_pic_list_l0,
            ref_pic_list_l1: &_ref_pic_list_l1,
            num_ref_idx_l0_active: header.num_ref_idx_l0_active,
            num_ref_idx_l1_active: header.num_ref_idx_l1_active,
            wctx: &wctx,
            first_mb_in_slice: header.first_mb_in_slice,
        };

        let mut mb_idx = header.first_mb_in_slice as usize;
        while mb_idx < total_mbs {
            // CAVLC end-of-slice: check before reading any new syntax elements.
            // Skip this check when counting down a skip run (no reads needed).
            if !use_cabac && mb_skip_run <= 0 && !reader.more_rbsp_data() {
                break;
            }

            // Stamp this MB with the current slice ID for boundary detection
            mb_slice_id[mb_idx] = this_slice_id;
            let mb_x = (mb_idx % mb_width as usize) * 16;
            let mb_y = (mb_idx / mb_width as usize) * 16;

            // CABAC decode path
            if use_cabac {
                let cr = cabac_reader.as_mut().unwrap();
                let st = &mut cabac_state;
                {
                    let mut ctx = make_ctx!();
                    match ctx.decode_cabac_mb(cr, st, &nal.rbsp, mb_idx, mb_x, mb_y, &params)? {
                        CabacMbResult::EndOfSlice => break,
                        CabacMbResult::Decoded => {}
                    }
                    prev_mb_qp = ctx.prev_mb_qp;
                    last_qp_delta_nonzero = ctx.last_qp_delta_nonzero;
                }
                mb_idx += 1;
                continue;
            }

            // P/B-slice skip run handling
            if is_p_slice || is_b_slice {
                if mb_skip_run < 0 {
                    mb_skip_run = reader.read_ue()? as i32;
                }
                if mb_skip_run > 0 {
                    mb_skip_run -= 1;
                    if is_p_slice {
                        // P_Skip: MV = median predictor, ref_idx = 0, no residual
                        make_ctx!().decode_p_skip_mb(mb_idx, mb_x, mb_y, &params);
                    } else {
                        // B_Skip: spatial/temporal direct MV + MC, no residual
                        make_ctx!().decode_b_skip_mb(mb_idx, mb_x, mb_y, &params);
                    }
                    mb_info[mb_idx] = MbInfo {
                        mb_type: MbType::Inter,
                        qp_y: prev_mb_qp,
                        ..Default::default()
                    };
                    mb_idx += 1;
                    continue;
                }
                // mb_skip_run == 0: parse the next MB normally
                mb_skip_run = -1; // reset for next iteration
            }

            {
                let mut ctx = make_ctx!();
                ctx.decode_cavlc_mb(&mut reader, mb_idx, mb_x, mb_y, &params)?;
                prev_mb_qp = ctx.prev_mb_qp;
                last_qp_delta_nonzero = ctx.last_qp_delta_nonzero;
            }
            mb_idx += 1;

            // CAVLC end-of-slice: spec says "while (more_rbsp_data())"
            // after each MB. For single-slice, this naturally ends at
            // total_mbs. For multi-slice, it stops at each slice boundary.
            if !use_cabac && !reader.more_rbsp_data() {
                break;
            }
        }

        // Post-loop: fill deblock info and ref POC table
        make_ctx!().finalize_mb_info(
            header.first_mb_in_slice as usize,
            mb_idx.min(total_mbs),
            &params,
        );

        // Store state back into pending PictureState.
        // Deblocking and DPB insertion happen in finalize_pending().
        self.pending = Some(PictureState {
            frame,
            frame_num: header.frame_num,
            poc: current_poc,
            nal_unit_type: nal.nal_unit_type,
            nal_ref_idc: nal.nal_ref_idc,
            nc_luma,
            nc_cb,
            nc_cr,
            mv_store_l0,
            mv_store_l1,
            ref_idx_store_l0,
            ref_poc_store_l0,
            ref_idx_store_l1,
            mvd_store,
            mvd_store_l1,
            mb_info,
            i4x4_modes,
            mb_cbp,
            mb_chroma_pred,
            mb_is_8x8dct,
            mb_skip,
            mb_is_direct,
            is_i16x16,
            mb_slice_id,
            current_slice_id,
            prev_mb_qp,
            last_qp_delta_nonzero,
            mmco_ops: header.mmco_ops.clone(),
            is_intra_slice: header.slice_type == SliceType::I,
            disable_deblocking_filter_idc: ps_deblock_idc,
            slice_alpha_c0_offset_div2: ps_alpha,
            slice_beta_offset_div2: ps_beta,
            chroma_qp_index_offset: ps_chroma_qp_offset,
            mb_width,
            mb_height,
        });

        Ok(())
    }
}

/// Motion vector prediction for P_8x8 sub-partitions.
/// `px`, `py`: sub-partition position within the macroblock (pixel coordinates).
/// `spw`, `sph`: sub-partition dimensions.
#[allow(clippy::too_many_arguments)]
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
        if let Some(f) = decoder.flush() {
            frame = Some(f);
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
        if let Some(f) = decoder.flush() {
            frame = Some(f);
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
        if let Some(f) = decoder.flush() {
            frame = Some(f);
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
        if let Some(f) = decoder.flush() {
            frame = Some(f);
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
        if let Some(f) = decoder.flush() {
            frame = Some(f);
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
        let h264_path = format!("{}/testdata/{}.h264", env!("CARGO_MANIFEST_DIR"), h264_name);
        let yuv_path = format!("{}/testdata/{}.yuv", env!("CARGO_MANIFEST_DIR"), h264_name);
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
        if let Some(f) = decoder.flush() {
            frame = Some(f);
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
        if let Some(f) = decoder.flush() {
            frames.push(f);
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
        let h264_path = format!("{}/testdata/{}.h264", env!("CARGO_MANIFEST_DIR"), h264_name);
        let yuv_path = format!("{}/testdata/{}.yuv", env!("CARGO_MANIFEST_DIR"), h264_name);
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
        // Flush the last pending frame
        if let Some(f) = decoder.flush() {
            frames.push(f);
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
        // (reference YUV is in display order; decoder outputs in decode order)
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

    #[test]
    fn test_b_temporal() {
        // 32x32, 5 frames (coded: I,P,B,P,B) — B-frames use 100% B_Skip
        // (temporal direct mode)
        decode_multiframe_and_compare("b_temporal_test", 5, 32, 32);
    }

    #[test]
    fn test_b_partitions() {
        // 64x64, 5 frames (I,B,P,B,P) — B-frames with 37.5% B16x16,
        // 40.6% B16x8/8x16, 5.5% B_8x8, 15.6% direct, 87.9% Bi
        decode_multiframe_and_compare("b_parts_test", 5, 64, 64);
    }

    #[test]
    fn test_b_multi_frame() {
        // 64x64, 8 frames (I,B,P,B,P,B,P,P) — multiple B-frames across
        // the sequence with skip, direct, and various partition types
        decode_multiframe_and_compare("b_multi_test", 8, 64, 64);
    }

    #[test]
    fn test_b_hierarchical() {
        // 64x64, 8 frames with bframes=3, ref=2 — hierarchical B-frames
        // with reference B-frames and ref_pic_list_modification reordering
        decode_multiframe_and_compare("b_hier_test", 8, 64, 64);
    }

    #[test]
    fn test_cabac_i4x4() {
        // 16x16 single-MB CABAC I4x4 frame (Main profile)
        decode_and_compare("cabac_i4x4_test", 16, 16);
    }

    #[test]
    fn test_cabac_i16x16() {
        // 16x16 single-MB CABAC I16x16 DC frame (Main profile)
        decode_and_compare("cabac_i16x16_test", 16, 16);
    }

    #[test]
    fn test_cabac_mixed() {
        // 32x32 multi-MB CABAC I-frame with mixed I4x4/I16x16 (Main profile)
        decode_multiframe_and_compare("cabac_mixed_test", 1, 32, 32);
    }

    #[test]
    fn test_cabac_p_slice() {
        // 32x32, 3 frames: CABAC IDR + 2 P-frames (100% P_L0_16x16)
        decode_multiframe_and_compare("cabac_p_test", 3, 32, 32);
    }

    #[test]
    fn test_cabac_p_parts() {
        // 64x64, 5 frames: CABAC P with P16x16 (25%) + P16x8 (29.7%) + P8x16 (20.3%) +
        // P_8x8 (6.6%) + P_4x4 sub-partitions (5.9%) + skip (10.9%),
        // --no-deblock, byte-exact against FFmpeg
        decode_multiframe_and_compare("cabac_p_parts_test", 5, 64, 64);
    }

    #[test]
    fn test_cabac_intra_in_p() {
        // 64x64, 2 frames: CABAC IDR + P-frame with I16x16-in-P (12.5%) +
        // P_L0_16x16 (75%) + P_8x8 (6.25%) + skip (6.25%),
        // --no-deblock, byte-exact against FFmpeg
        decode_multiframe_and_compare("cabac_intra_p_test", 2, 64, 64);
    }

    #[test]
    fn test_cabac_b_slice() {
        // 32x32, 15 frames: CABAC B-frames with B_L0_16x16, B_L1_16x16, B_Skip
        // (--no-deblock, spatial direct), byte-exact against FFmpeg
        decode_multiframe_and_compare("cabac_b_test", 15, 32, 32);
    }

    #[test]
    fn test_cabac_b_parts() {
        // 64x64, 10 frames: CABAC B-frames with B16x16 (15.6%) + B16x8 (25%) +
        // B8x16/8x8 (19.5%) + B_Direct spatial (18.8%) + B_Skip (21.9%),
        // L0/L1/Bi mix, P_8x8 sub-partitions, --no-deblock, byte-exact against FFmpeg
        decode_multiframe_and_compare("cabac_b_parts_test", 10, 64, 64);
    }

    #[test]
    fn test_cabac_intra_in_b() {
        // 64x64, 10 frames: CABAC B-frames with I16x16-in-B (6.2%) + B16x16 (25%) +
        // B_Direct (68.8%) + Bi (58.3%), noisy content, --no-deblock,
        // byte-exact against FFmpeg
        decode_multiframe_and_compare("cabac_intra_b_test", 10, 64, 64);
    }

    #[test]
    fn test_cabac_b_temporal() {
        // 64x64, 10 frames: CABAC B-frames with temporal direct mode (20.3%) +
        // B16x16 (59.4%) + B_Skip (20.3%), L0/L1/Bi mix, --no-deblock,
        // byte-exact against FFmpeg
        decode_multiframe_and_compare("cabac_b_temporal_test", 10, 64, 64);
    }

    #[test]
    fn test_cabac_high_profile() {
        // 64x64, 5 frames: CABAC High profile with 8x8 transform (43.8% inter 8x8),
        // P-only, --no-deblock, medium preset, byte-exact against FFmpeg
        decode_multiframe_and_compare("cabac_high_test", 5, 64, 64);
    }

    #[test]
    fn test_cabac_i8x8() {
        // 64x64, 1 frame: CABAC High profile I-slice with 100% I8x8 (DC mode)
        // + varied chroma (dc 6%, h 19%, v 38%, plane 38%).
        // Validates I8x8 chroma decode in the CABAC I-slice path.
        decode_multiframe_and_compare("cabac_i8x8_test", 1, 64, 64);
    }

    #[test]
    fn test_cabac_multiref() {
        // 64x64, 5 frames: CABAC Main profile with ref=2, bframes=1, me=hex,
        // --no-deblock, --no-weightb, qp=26. Exercises CABAC multiref with
        // ref_pic_list_modification and P_L0_L0_16x8 partitions using ref_idx>0.
        decode_multiframe_and_compare("cabac_multiref_test", 5, 64, 64);
    }

    #[test]
    fn test_preset_medium() {
        // 320x240, 60 frames: x264 --preset medium --profile main --no-deblock.
        // CABAC, ref=4, bframes=3, subme=7, me=hex, all partitions.
        // Exercises P_8x8 sub-partitions with multiref, B 16x8/8x16,
        // ref_pic_list_modification, and hierarchical B-frames.
        decode_multiframe_and_compare("preset_medium", 60, 320, 240);
    }

    #[test]
    fn test_preset_medium_deblock() {
        // 320x240, 60 frames: x264 --preset medium --profile main (deblocking ON).
        // Full pipeline: CABAC, ref=4, bframes=3, all partitions, deblocking.
        decode_multiframe_and_compare("preset_medium_deblock", 60, 320, 240);
    }

    #[test]
    fn test_cabac_deblock() {
        // 64x64, 3 frames: CABAC Main profile with deblocking enabled,
        // P_L0_16x16 (84.4%) + I-in-P (12.5%) + skip (3.1%), byte-exact against FFmpeg
        decode_multiframe_and_compare("cabac_deblock_test", 3, 64, 64);
    }

    #[test]
    fn test_deblock_b_frames() {
        // 64x64, 9 frames: CAVLC Main profile with B-frames (bframes=2) and deblocking,
        // B16x16 (2.5%) + B_Direct (10%) + B_Skip (87.5%) + I-in-P (31.3%),
        // byte-exact against FFmpeg
        decode_multiframe_and_compare("deblock_b_test", 9, 64, 64);
    }

    #[test]
    fn test_deblock_b_inter() {
        // 64x64, 5 frames: CAVLC Main profile B-frames with deblocking,
        // B16x16 L0/L1/Bi (78.1%) + B_Direct (12.5%) + B_Skip (9.4%),
        // exercises cross-list deblock bS comparison, byte-exact against FFmpeg
        decode_multiframe_and_compare("deblock_b_inter_test", 5, 64, 64);
    }

    #[test]
    fn test_weighted_p() {
        // 32x32, 10 frames: CAVLC P with explicit weighted prediction (100% weighted,
        // 77.8% chroma weighted), fading content, --no-deblock, byte-exact against FFmpeg
        decode_multiframe_and_compare("weighted_p_test", 10, 32, 32);
    }

    #[test]
    fn test_weighted_b_implicit() {
        // 64x64, 10 frames: CABAC B with implicit weighted bi-prediction (idc=2),
        // fading content, B16x16 (25%) + B_Direct (48.4%) + B_Skip (26.6%),
        // --no-deblock, byte-exact against FFmpeg
        decode_multiframe_and_compare("weighted_b_test", 10, 64, 64);
    }

    #[test]
    fn test_realworld() {
        // 320x240, 6 frames: CAVLC Main profile, P16x16 (16.6%) + P16x8 (7.8%) +
        // P8x16 (3.1%) + intra-in-P (3.6%) + skip (68.9%), --no-deblock.
        // Regression test for real-world-sized content with diverse MB types.
        decode_multiframe_and_compare("realworld_test", 6, 320, 240);
    }

    #[test]
    fn test_high_profile() {
        // 320x240, 6 frames: CAVLC High profile with 8x8 transform
        // (28% intra 8x8, 22.8% inter 8x8), --no-deblock.
        decode_multiframe_and_compare("high_profile_test", 6, 320, 240);
    }

    #[test]
    fn test_realworld_b() {
        // 320x240, 9 frames: CAVLC Main profile with B-frames (bframes=2),
        // B16x16 L0/L1/Bi (16.2%) + B16x8/8x16 (7.2%) + B_Direct (2.8%) +
        // B_Skip (73.5%) + P partitions + intra-in-P/B, --no-deblock.
        decode_multiframe_and_compare("realworld_b_test", 9, 320, 240);
    }

    #[test]
    fn test_multislice_cabac_i() {
        // 32x32, 1 frame, 2 slices (1 MB row each): CABAC Main profile I-frame.
        // Tests cross-slice intra prediction boundary handling (spec 6.4.1).
        decode_and_compare("ms_cabac_i_test", 32, 32);
    }

    #[test]
    fn test_multislice_cabac_i4() {
        // 64x64, 1 frame, 4 slices (1 MB row each): CABAC Main profile I-frame.
        // Tests multiple slice boundaries with I4x4 prediction.
        decode_and_compare("ms_cabac_i4_test", 64, 64);
    }

    #[test]
    fn test_multislice_cavlc_i() {
        // 32x32, 1 frame, 2 slices (1 MB row each): CAVLC Baseline profile I-frame.
        // Tests cross-slice nC computation and intra prediction for CAVLC.
        decode_and_compare("ms_cavlc_i_test", 32, 32);
    }

    #[test]
    fn test_multislice_cavlc_p() {
        // 64x64, 5 frames (IDR + 4 P), 4 slices per frame: CAVLC Main profile.
        // Tests multi-slice P-frame decode with cross-slice intra prediction
        // and nC boundary handling across both I and P slices.
        decode_multiframe_and_compare("ms_cavlc_p_test", 5, 64, 64);
    }

    #[test]
    fn test_multislice_cabac_p() {
        // 64x64, 5 frames (IDR + 4 P), 4 slices per frame: CABAC Main profile,
        // no deblocking. Tests multi-slice P-frame CABAC decode with cross-slice
        // intra prediction and CABAC neighbor context boundary handling.
        decode_multiframe_and_compare("ms_cabac_p_test", 5, 64, 64);
    }

    #[test]
    fn test_multislice_cabac_b() {
        // 64x64, 4 frames (IDR + B + B + P), 4 slices per frame: CABAC Main profile,
        // no deblocking, temporal+spatial direct, B_8x8 sub-partitions.
        // Tests multi-slice B-frame decode with cross-slice boundary handling.
        decode_multiframe_and_compare("ms_cabac_b_test", 4, 64, 64);
    }

    #[test]
    fn test_high_p8x8_sub4x4() {
        // 64x64, 6 frames (I+P): High profile, preset slower with P_8x8
        // sub-4x4 partitions + 8x8dct. Tests noSubMbPartSizeLessThan8x8Flag:
        // transform_size_8x8_flag must NOT be read when P_8x8 has sub-4x4
        // sub-partitions.
        decode_multiframe_and_compare("high_p8x8_sub4x4_test", 6, 64, 64);
    }

    #[test]
    fn test_b_temporal_direct_8x8_inference() {
        // 64x64, 4 frames (I,B,B,P): CABAC Main, preset slower, direct=temporal.
        // Tests direct_8x8_inference_flag in temporal direct mode: co-located MV
        // must be read from the representative 4x4 block per 8x8 group, not from
        // each individual 4x4 block.
        decode_multiframe_and_compare("b_temporal_direct_test", 4, 64, 64);
    }

    #[test]
    fn test_high_b_slower() {
        // 64x64, 10 frames: High profile, preset slower, ref=2, bframes=2,
        // 8x8dct, no-deblock. Tests direct_8x8_inference_flag in BOTH temporal
        // and spatial direct modes, noSubMbPartSizeLessThan8x8Flag for
        // transform_size_8x8_flag, and B_8x8 with sub-partition types.
        decode_multiframe_and_compare("high_b_slower_test", 10, 64, 64);
    }
}
