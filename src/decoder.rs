use std::collections::HashMap;

use std::rc::Rc;

use crate::bitstream::BitstreamReader;
use crate::cavlc::parse_residual_block_cavlc;
use crate::deblock::{self, MbInfo, MbType};
use crate::dpb::{DecodedPicture, Dpb, ReferenceStatus};
use crate::error::DecodeError;
use crate::inter_pred;
use crate::intra_pred::{
    predict_chroma_8x8, predict_intra_4x4, predict_intra_8x8,
};
use crate::mv_pred::{predict_mv, predict_mv_sub, ref_pic_safe, WeightContext};
use crate::nal::{NalUnit, NalUnitType};
use crate::neighbor::{
    cabac_amvd, cabac_neighbor_nz_chroma, cabac_neighbor_nz_luma, cabac_neighbor_ref, compute_nc,
    dequant_4x4_ac_raster, predict_i4x4_mode,
};
use crate::pps::{parse_pps, Pps};
use crate::residual::{
    chroma_qp, dequant_4x4_full, dequant_8x8, dequant_chroma_dc, dequant_luma_dc_i16x16,
    inverse_dct_4x4, inverse_dct_8x8, inverse_hadamard_2x2, inverse_hadamard_4x4,
    BLOCK_INDEX_TO_OFFSET, CBP_INTER_TABLE, CBP_INTRA_TABLE, ZIGZAG_4X4, ZIGZAG_8X8_CABAC,
    ZIGZAG_8X8_CAVLC,
};
use crate::slice::{parse_slice_header, SliceType};
use crate::slice_context::SliceContext;
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

                // End-of-slice check via terminate (not for the first MB)
                if mb_idx > header.first_mb_in_slice as usize && cr.get_cabac_terminate() != 0 {
                    break;
                }
                // P/B-slice CABAC path
                if is_p_slice || is_b_slice {
                    // Decode skip flag
                    // Unavailable neighbors are treated as skipped (ctx not incremented)
                    let left_skip = if !mb_idx.is_multiple_of(mb_width as usize)
                        && mb_slice_id[mb_idx - 1] == this_slice_id
                    {
                        mb_skip[mb_idx - 1]
                    } else {
                        true
                    };
                    let top_skip = if mb_idx >= mb_width as usize
                        && mb_slice_id[mb_idx - mb_width as usize] == this_slice_id
                    {
                        mb_skip[mb_idx - mb_width as usize]
                    } else {
                        true
                    };
                    let is_skip = cr.decode_mb_skip(st, left_skip, top_skip, is_b_slice);
                    if is_skip {
                        mb_skip[mb_idx] = true;
                        if is_p_slice {
                            // P_Skip: median MV, ref=0, no residual
                            make_ctx!().decode_p_skip_mb(
                                mb_idx,
                                mb_x,
                                mb_y,
                                &ref_pic_list,
                                &wctx,
                                use_weight,
                            );
                        } else {
                            // B_Skip: spatial/temporal direct MV + MC, no residual
                            make_ctx!().decode_b_skip_mb(
                                mb_idx,
                                mb_x,
                                mb_y,
                                header.direct_spatial_mv_pred_flag,
                                sps.direct_8x8_inference_flag,
                                current_poc,
                                &_ref_pic_list_l0,
                                &_ref_pic_list_l1,
                                &wctx,
                                use_weight,
                            );
                        }
                        mb_info[mb_idx] = MbInfo {
                            mb_type: MbType::Inter,
                            qp_y: prev_mb_qp,
                            ..Default::default()
                        };
                        last_qp_delta_nonzero = false;
                        mb_idx += 1;
                        continue;
                    }

                    // Non-skip: decode mb_type
                    let raw_mb_type = if is_p_slice {
                        cr.decode_p_mb_type(st)
                    } else {
                        let left_not_direct = if !mb_idx.is_multiple_of(mb_width as usize)
                            && mb_slice_id[mb_idx - 1] == this_slice_id
                        {
                            !mb_is_direct[mb_idx - 1]
                        } else {
                            false
                        };
                        let top_not_direct = if mb_idx >= mb_width as usize
                            && mb_slice_id[mb_idx - mb_width as usize] == this_slice_id
                        {
                            !mb_is_direct[mb_idx - mb_width as usize]
                        } else {
                            false
                        };
                        cr.decode_b_mb_type(st, left_not_direct, top_not_direct)
                    };

                    // Check if it's intra-in-P/B
                    let intra_limit = if is_p_slice { 5 } else { 23 };
                    if raw_mb_type >= intra_limit {
                        // Intra in P/B slice — decode as I-slice mb_type
                        // The mb_type already includes the intra type from decode_intra_mb_type
                        let i_mb_type = raw_mb_type - intra_limit;

                        // Set ref_idx to -1 for all blocks (intra MB)
                        for blk in 0..16 {
                            ref_idx_store_l0[mb_idx * 16 + blk] = -1;
                            ref_idx_store_l1[mb_idx * 16 + blk] = -1;
                        }

                        // Cross-slice intra prediction: neighbors from other slices unavailable (spec 6.4.1)
                        let above_mb_avail = mb_idx >= mb_width as usize
                            && mb_slice_id[mb_idx - mb_width as usize] == this_slice_id;
                        let left_mb_avail = !mb_idx.is_multiple_of(mb_width as usize)
                            && mb_slice_id[mb_idx - 1] == this_slice_id;
                        let above_left_mb_avail = mb_idx >= mb_width as usize
                            && !mb_idx.is_multiple_of(mb_width as usize)
                            && mb_slice_id[mb_idx - mb_width as usize - 1] == this_slice_id;
                        let above_right_mb_avail = mb_idx >= mb_width as usize
                            && (mb_idx % mb_width as usize) + 1 < mb_width as usize
                            && mb_slice_id[mb_idx - mb_width as usize + 1] == this_slice_id;

                        if i_mb_type == 0 {
                            // I4x4/I8x8 in P/B
                            let nts = {
                                let left = if !mb_idx.is_multiple_of(mb_width as usize)
                                    && mb_slice_id[mb_idx - 1] == this_slice_id
                                {
                                    mb_is_8x8dct[mb_idx - 1] as usize
                                } else {
                                    0
                                };
                                let top = if mb_idx >= mb_width as usize
                                    && mb_slice_id[mb_idx - mb_width as usize] == this_slice_id
                                {
                                    mb_is_8x8dct[mb_idx - mb_width as usize] as usize
                                } else {
                                    0
                                };
                                left + top
                            };
                            let use_8x8_intra_pb = pps.transform_8x8_mode_flag
                                && cr.get_cabac(&mut st[399 + nts]) != 0;
                            mb_is_8x8dct[mb_idx] = use_8x8_intra_pb;

                            let num_modes = if use_8x8_intra_pb { 4 } else { 16 };
                            let mut pred_modes = [2u8; 16];
                            for blk_idx in 0..num_modes {
                                let blk = if use_8x8_intra_pb {
                                    blk_idx * 4
                                } else {
                                    blk_idx
                                };
                                let predicted = predict_i4x4_mode(
                                    &i4x4_modes,
                                    mb_idx,
                                    mb_width as usize,
                                    blk,
                                    &mb_slice_id,
                                    this_slice_id,
                                );
                                let mode = cr.decode_intra4x4_pred_mode(st, predicted);
                                if use_8x8_intra_pb {
                                    for sub in 0..4 {
                                        pred_modes[blk_idx * 4 + sub] = mode;
                                        i4x4_modes[mb_idx * 16 + blk_idx * 4 + sub] = mode;
                                    }
                                } else {
                                    pred_modes[blk] = mode;
                                    i4x4_modes[mb_idx * 16 + blk] = mode;
                                }
                            }
                            let left_cm = if !mb_idx.is_multiple_of(mb_width as usize)
                                && mb_slice_id[mb_idx - 1] == this_slice_id
                            {
                                mb_chroma_pred[mb_idx - 1]
                            } else {
                                0
                            };
                            let top_cm = if mb_idx >= mb_width as usize
                                && mb_slice_id[mb_idx - mb_width as usize] == this_slice_id
                            {
                                mb_chroma_pred[mb_idx - mb_width as usize]
                            } else {
                                0
                            };
                            let intra_chroma_pred_mode =
                                cr.decode_chroma_pred_mode(st, left_cm, top_cm);
                            mb_chroma_pred[mb_idx] = intra_chroma_pred_mode;

                            let unavail_cbp: u16 = 0x00F; // inter unavailable default
                            let left_cbp_raw = if !mb_idx.is_multiple_of(mb_width as usize)
                                && mb_slice_id[mb_idx - 1] == this_slice_id
                            {
                                mb_cbp[mb_idx - 1]
                            } else {
                                unavail_cbp
                            };
                            let top_cbp_raw = if mb_idx >= mb_width as usize
                                && mb_slice_id[mb_idx - mb_width as usize] == this_slice_id
                            {
                                mb_cbp[mb_idx - mb_width as usize]
                            } else {
                                unavail_cbp
                            };
                            let left_cbp = ((left_cbp_raw & 0x7F0)
                                | (left_cbp_raw & 2)
                                | (((left_cbp_raw >> 2) & 2) << 2))
                                as u8;
                            let top_cbp = top_cbp_raw as u8;
                            let cbp_luma = cr.decode_cbp_luma(st, left_cbp, top_cbp);
                            let left_cbp_c = ((left_cbp_raw >> 4) & 3) as u8;
                            let top_cbp_c = ((top_cbp_raw >> 4) & 3) as u8;
                            let cbp_chroma = cr.decode_cbp_chroma(st, left_cbp_c, top_cbp_c);
                            mb_cbp[mb_idx] = (cbp_luma as u16) | ((cbp_chroma as u16) << 4);

                            let qp_y = if cbp_luma != 0 || cbp_chroma != 0 {
                                let delta = cr.decode_mb_qp_delta(st, last_qp_delta_nonzero);
                                last_qp_delta_nonzero = delta != 0;
                                ((prev_mb_qp + delta + 52) % 52 + 52) % 52
                            } else {
                                last_qp_delta_nonzero = false;
                                prev_mb_qp
                            };
                            prev_mb_qp = qp_y;
                            let _qp_c = chroma_qp(qp_y, pps.chroma_qp_index_offset);

                            // Decode luma residual + prediction
                            if use_8x8_intra_pb {
                                // I8x8 in P/B: same as I-slice I8x8 CABAC path
                                let mut luma_residual = [0i32; 256];
                                for i8x8 in 0..4 {
                                    if cbp_luma & (1 << i8x8) != 0 {
                                        // cat=5: no coded_block_flag, CBP bit is sufficient
                                        let (coeffs, tc) = cr.decode_residual_cabac(st, 5, 64);
                                        let tc_per = tc.div_ceil(4);
                                        for sub in 0..4 {
                                            nc_luma[mb_idx * 16 + i8x8 * 4 + sub] = tc_per;
                                        }
                                        let mut block_8x8 = [0i32; 64];
                                        for (pos, val) in &coeffs {
                                            block_8x8[ZIGZAG_8X8_CABAC[*pos]] = *val;
                                        }
                                        dequant_8x8(&mut block_8x8, qp_y, &pps.scaling_list_8x8[0]);
                                        inverse_dct_8x8(&mut block_8x8);
                                        let row_off = (i8x8 / 2) * 8;
                                        let col_off = (i8x8 % 2) * 8;
                                        for r in 0..8 {
                                            for c in 0..8 {
                                                luma_residual[(row_off + r) * 16 + col_off + c] =
                                                    block_8x8[r * 8 + c];
                                            }
                                        }
                                    }
                                }
                                for i8x8 in 0..4 {
                                    let row_off = (i8x8 / 2) * 8;
                                    let col_off = (i8x8 % 2) * 8;
                                    let px = mb_x + col_off;
                                    let py = mb_y + row_off;
                                    let above_avail_8 = py > 0 && (row_off > 0 || above_mb_avail);
                                    let above_buf: Option<[u8; 16]> = if above_avail_8 {
                                        let mut buf = [0u8; 16];
                                        for (i, b) in buf.iter_mut().enumerate().take(8) {
                                            *b = frame.y[(py - 1) * stride + px + i];
                                        }
                                        let has_tr = if row_off == 0 {
                                            if px + 8 < (mb_x + 16).min(stride) {
                                                true
                                            } else if px + 8 < stride {
                                                above_right_mb_avail
                                            } else {
                                                false
                                            }
                                        } else {
                                            col_off == 0
                                        };
                                        if has_tr {
                                            for i in 0..8 {
                                                buf[8 + i] = frame.y[(py - 1) * stride
                                                    + (px + 8 + i).min(stride - 1)];
                                            }
                                        } else {
                                            let last = buf[7];
                                            buf[8..].fill(last);
                                        }
                                        Some(buf)
                                    } else {
                                        None
                                    };
                                    let left_avail_8 = px > 0 && (col_off > 0 || left_mb_avail);
                                    let left_buf: Option<[u8; 8]> = if left_avail_8 {
                                        let mut buf = [0u8; 8];
                                        for (i, b) in buf.iter_mut().enumerate() {
                                            *b = frame.y[(py + i) * stride + px - 1];
                                        }
                                        Some(buf)
                                    } else {
                                        None
                                    };
                                    let al_avail_8 = px > 0
                                        && py > 0
                                        && ((row_off > 0 && col_off > 0)
                                            || (row_off > 0 && col_off == 0 && left_mb_avail)
                                            || (row_off == 0 && col_off > 0 && above_mb_avail)
                                            || (row_off == 0
                                                && col_off == 0
                                                && above_left_mb_avail));
                                    let al = if al_avail_8 {
                                        Some(frame.y[(py - 1) * stride + px - 1])
                                    } else {
                                        None
                                    };
                                    let has_tr = above_buf.is_some()
                                        && ((row_off == 0 && px + 8 < stride)
                                            || (row_off > 0 && col_off == 0));
                                    let mut pred = [0u8; 64];
                                    predict_intra_8x8(
                                        pred_modes[i8x8 * 4],
                                        above_buf.as_ref().map(|b| &b[..]),
                                        left_buf.as_ref().map(|b| &b[..]),
                                        al,
                                        has_tr,
                                        &mut pred,
                                    );
                                    for r in 0..8 {
                                        for c in 0..8 {
                                            let val = (pred[r * 8 + c] as i32
                                                + luma_residual[(row_off + r) * 16 + col_off + c])
                                                .clamp(0, 255)
                                                as u8;
                                            frame.y[(py + r) * stride + px + c] = val;
                                        }
                                    }
                                }
                            } else {
                                // I4x4 path (existing)
                                for blk in 0..16 {
                                    let (blk_row, blk_col) = BLOCK_INDEX_TO_OFFSET[blk];
                                    let px = mb_x + blk_col;
                                    let py = mb_y + blk_row;
                                    let mut block_coeffs = [0i32; 16];
                                    if cbp_luma & (1 << (blk / 4)) != 0 {
                                        let left_nz = cabac_neighbor_nz_luma(
                                            &nc_luma,
                                            mb_idx,
                                            mb_width as usize,
                                            blk,
                                            true,
                                            true,
                                            &mb_slice_id,
                                            this_slice_id,
                                        );
                                        let top_nz = cabac_neighbor_nz_luma(
                                            &nc_luma,
                                            mb_idx,
                                            mb_width as usize,
                                            blk,
                                            false,
                                            true,
                                            &mb_slice_id,
                                            this_slice_id,
                                        );
                                        if cr.decode_coded_block_flag(st, 2, left_nz, top_nz) {
                                            let (coeffs, tc) = cr.decode_residual_cabac(st, 2, 16);
                                            nc_luma[mb_idx * 16 + blk] = tc;
                                            for (pos, val) in &coeffs {
                                                let (r, c) = ZIGZAG_4X4[*pos];
                                                block_coeffs[r * 4 + c] = *val;
                                            }
                                            dequant_4x4_full(
                                                &mut block_coeffs,
                                                qp_y,
                                                &pps.scaling_list_4x4[0],
                                            );
                                        }
                                    }
                                    inverse_dct_4x4(&mut block_coeffs);

                                    let local_row = py - mb_y;
                                    let local_col = px - mb_x;
                                    let above_avail = py > 0 && (local_row > 0 || above_mb_avail);
                                    let above_buf: Option<[u8; 8]> = if above_avail {
                                        let mut buf = [0u8; 8];
                                        for (i, b) in buf.iter_mut().enumerate().take(4) {
                                            *b = frame.y[(py - 1) * stride + px + i];
                                        }
                                        let topright_avail = if local_row == 0 {
                                            if px + 4 < (mb_x + 16).min(stride) {
                                                true
                                            } else if px + 4 < stride {
                                                above_right_mb_avail
                                            } else {
                                                false
                                            }
                                        } else {
                                            !matches!(blk, 3 | 7 | 11 | 13 | 15)
                                        };
                                        if topright_avail {
                                            for (i, b) in buf.iter_mut().enumerate().skip(4) {
                                                let col = (px + i).min(stride - 1);
                                                *b = frame.y[(py - 1) * stride + col];
                                            }
                                        } else {
                                            let last = buf[3];
                                            buf[4..8].fill(last);
                                        }
                                        Some(buf)
                                    } else {
                                        None
                                    };
                                    let left_avail = px > 0 && (local_col > 0 || left_mb_avail);
                                    let left_buf: Option<[u8; 4]> = if left_avail {
                                        let mut buf = [0u8; 4];
                                        for (i, b) in buf.iter_mut().enumerate() {
                                            *b = frame.y[(py + i) * stride + px - 1];
                                        }
                                        Some(buf)
                                    } else {
                                        None
                                    };
                                    let al_avail = px > 0
                                        && py > 0
                                        && ((local_row > 0 && local_col > 0)
                                            || (local_row > 0 && local_col == 0 && left_mb_avail)
                                            || (local_row == 0 && local_col > 0 && above_mb_avail)
                                            || (local_row == 0
                                                && local_col == 0
                                                && above_left_mb_avail));
                                    let above_left_val = if al_avail {
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
                                    for r in 0..4 {
                                        for c in 0..4 {
                                            let val = (pred[r * 4 + c] as i32
                                                + block_coeffs[r * 4 + c])
                                                .clamp(0, 255)
                                                as u8;
                                            frame.y[(py + r) * stride + px + c] = val;
                                        }
                                    }
                                }
                            } // close if use_8x8_intra_pb else

                            // Chroma prediction
                            let chroma_width = (width / 2) as usize;
                            let chroma_mb_x = mb_x / 2;
                            let chroma_mb_y = mb_y / 2;
                            let (pred_u, pred_v) = make_ctx!().predict_chroma_intra(
                                mb_x,
                                mb_y,
                                intra_chroma_pred_mode,
                                above_mb_avail,
                                left_mb_avail,
                                above_left_mb_avail,
                            );

                            // Chroma DC residual
                            let mut chroma_dc_cb = [0i32; 4];
                            let mut chroma_dc_cr = [0i32; 4];
                            if cbp_chroma >= 1 {
                                let left_dc_nz = if !mb_idx.is_multiple_of(mb_width as usize)
                                    && mb_slice_id[mb_idx - 1] == this_slice_id
                                {
                                    (mb_cbp[mb_idx - 1] >> 6) & 1 != 0
                                } else {
                                    true
                                };
                                let top_dc_nz = if mb_idx >= mb_width as usize
                                    && mb_slice_id[mb_idx - mb_width as usize] == this_slice_id
                                {
                                    (mb_cbp[mb_idx - mb_width as usize] >> 6) & 1 != 0
                                } else {
                                    true
                                };
                                if cr.decode_coded_block_flag(st, 3, left_dc_nz, top_dc_nz) {
                                    let (coeffs, _) = cr.decode_residual_cabac(st, 3, 4);
                                    for (p, v) in coeffs {
                                        chroma_dc_cb[p] = v;
                                    }
                                    mb_cbp[mb_idx] |= 0x40;
                                }

                                let left_dc_nz_cr = if !mb_idx.is_multiple_of(mb_width as usize)
                                    && mb_slice_id[mb_idx - 1] == this_slice_id
                                {
                                    (mb_cbp[mb_idx - 1] >> 7) & 1 != 0
                                } else {
                                    true
                                };
                                let top_dc_nz_cr = if mb_idx >= mb_width as usize
                                    && mb_slice_id[mb_idx - mb_width as usize] == this_slice_id
                                {
                                    (mb_cbp[mb_idx - mb_width as usize] >> 7) & 1 != 0
                                } else {
                                    true
                                };
                                if cr.decode_coded_block_flag(st, 3, left_dc_nz_cr, top_dc_nz_cr) {
                                    let (coeffs, _) = cr.decode_residual_cabac(st, 3, 4);
                                    for (p, v) in coeffs {
                                        chroma_dc_cr[p] = v;
                                    }
                                    mb_cbp[mb_idx] |= 0x80;
                                }
                            }

                            // Chroma AC residual
                            let mut chroma_ac_cb = [[0i32; 15]; 4];
                            let mut chroma_ac_cr = [[0i32; 15]; 4];
                            if cbp_chroma >= 2 {
                                for blk in 0..4 {
                                    let left_nz = cabac_neighbor_nz_chroma(
                                        &nc_cb,
                                        mb_idx,
                                        mb_width as usize,
                                        blk,
                                        true,
                                        true,
                                        &mb_slice_id,
                                        this_slice_id,
                                    );
                                    let top_nz = cabac_neighbor_nz_chroma(
                                        &nc_cb,
                                        mb_idx,
                                        mb_width as usize,
                                        blk,
                                        false,
                                        true,
                                        &mb_slice_id,
                                        this_slice_id,
                                    );
                                    if cr.decode_coded_block_flag(st, 4, left_nz, top_nz) {
                                        let (coeffs, tc) = cr.decode_residual_cabac(st, 4, 15);
                                        nc_cb[mb_idx * 4 + blk] = tc;
                                        for (p, v) in coeffs {
                                            chroma_ac_cb[blk][p] = v;
                                        }
                                    }
                                }
                                for blk in 0..4 {
                                    let left_nz = cabac_neighbor_nz_chroma(
                                        &nc_cr,
                                        mb_idx,
                                        mb_width as usize,
                                        blk,
                                        true,
                                        true,
                                        &mb_slice_id,
                                        this_slice_id,
                                    );
                                    let top_nz = cabac_neighbor_nz_chroma(
                                        &nc_cr,
                                        mb_idx,
                                        mb_width as usize,
                                        blk,
                                        false,
                                        true,
                                        &mb_slice_id,
                                        this_slice_id,
                                    );
                                    if cr.decode_coded_block_flag(st, 4, left_nz, top_nz) {
                                        let (coeffs, tc) = cr.decode_residual_cabac(st, 4, 15);
                                        nc_cr[mb_idx * 4 + blk] = tc;
                                        for (p, v) in coeffs {
                                            chroma_ac_cr[blk][p] = v;
                                        }
                                    }
                                }
                            }

                            // Chroma reconstruction
                            for (plane_dc, plane_ac, pred_plane, frame_plane, scale_idx) in [
                                (
                                    &mut chroma_dc_cb,
                                    &chroma_ac_cb,
                                    &pred_u,
                                    &mut frame.u,
                                    1usize,
                                ),
                                (
                                    &mut chroma_dc_cr,
                                    &chroma_ac_cr,
                                    &pred_v,
                                    &mut frame.v,
                                    2usize,
                                ),
                            ] {
                                let chroma_scale = &pps.scaling_list_4x4[scale_idx];
                                if cbp_chroma >= 1 {
                                    inverse_hadamard_2x2(plane_dc);
                                    dequant_chroma_dc(plane_dc, _qp_c, chroma_scale[0]);
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
                                        dequant_4x4_ac_raster(
                                            &mut block_raster,
                                            _qp_c,
                                            chroma_scale,
                                        );
                                    }
                                    inverse_dct_4x4(&mut block_raster);
                                    for r in 0..4 {
                                        for c in 0..4 {
                                            chroma_residual[(blk_row + r) * 8 + blk_col + c] =
                                                block_raster[r * 4 + c];
                                        }
                                    }
                                }
                                for y in 0..8 {
                                    for x in 0..8 {
                                        let val = (pred_plane[y * 8 + x] as i32
                                            + chroma_residual[y * 8 + x])
                                            .clamp(0, 255)
                                            as u8;
                                        frame_plane
                                            [(chroma_mb_y + y) * chroma_width + chroma_mb_x + x] =
                                            val;
                                    }
                                }
                            }
                            mb_info[mb_idx] = MbInfo {
                                mb_type: MbType::Intra,
                                qp_y,
                                ..Default::default()
                            };
                        } else if i_mb_type <= 24 {
                            // I16x16 in P/B
                            is_i16x16[mb_idx] = true;
                            let mt = i_mb_type - 1;
                            let i16_pred = (mt % 4) as u8;
                            let cbp_chroma = ((mt / 4) % 3) as u8;
                            let cbp_luma = if mt >= 12 { 15u8 } else { 0u8 };
                            let left_cm = if !mb_idx.is_multiple_of(mb_width as usize)
                                && mb_slice_id[mb_idx - 1] == this_slice_id
                            {
                                mb_chroma_pred[mb_idx - 1]
                            } else {
                                0
                            };
                            let top_cm = if mb_idx >= mb_width as usize
                                && mb_slice_id[mb_idx - mb_width as usize] == this_slice_id
                            {
                                mb_chroma_pred[mb_idx - mb_width as usize]
                            } else {
                                0
                            };
                            let intra_chroma_pred_mode =
                                cr.decode_chroma_pred_mode(st, left_cm, top_cm);
                            mb_chroma_pred[mb_idx] = intra_chroma_pred_mode;

                            let delta = cr.decode_mb_qp_delta(st, last_qp_delta_nonzero);
                            last_qp_delta_nonzero = delta != 0;
                            let qp_y = ((prev_mb_qp + delta + 52) % 52 + 52) % 52;
                            prev_mb_qp = qp_y;
                            let qp_c = chroma_qp(qp_y, pps.chroma_qp_index_offset);

                            // Luma DC (cat=0, 16 coefficients)
                            let mut luma_dc = [0i32; 16];
                            let dc_left_nz = if !mb_idx.is_multiple_of(mb_width as usize)
                                && mb_slice_id[mb_idx - 1] == this_slice_id
                            {
                                (mb_cbp[mb_idx - 1] >> 8) & 1 != 0
                            } else {
                                true
                            };
                            let dc_top_nz = if mb_idx >= mb_width as usize
                                && mb_slice_id[mb_idx - mb_width as usize] == this_slice_id
                            {
                                (mb_cbp[mb_idx - mb_width as usize] >> 8) & 1 != 0
                            } else {
                                true
                            };
                            if cr.decode_coded_block_flag(st, 0, dc_left_nz, dc_top_nz) {
                                let (coeffs, _) = cr.decode_residual_cabac(st, 0, 16);
                                for (p, v) in coeffs {
                                    luma_dc[p] = v;
                                }
                                mb_cbp[mb_idx] |= 0x100;
                            }

                            // Luma AC (cat=1, 15 coefficients per block)
                            let mut luma_ac = [[0i32; 15]; 16];
                            if cbp_luma != 0 {
                                for blk in 0..16 {
                                    let left_nz = cabac_neighbor_nz_luma(
                                        &nc_luma,
                                        mb_idx,
                                        mb_width as usize,
                                        blk,
                                        true,
                                        true,
                                        &mb_slice_id,
                                        this_slice_id,
                                    );
                                    let top_nz = cabac_neighbor_nz_luma(
                                        &nc_luma,
                                        mb_idx,
                                        mb_width as usize,
                                        blk,
                                        false,
                                        true,
                                        &mb_slice_id,
                                        this_slice_id,
                                    );
                                    if cr.decode_coded_block_flag(st, 1, left_nz, top_nz) {
                                        let (coeffs, tc) = cr.decode_residual_cabac(st, 1, 15);
                                        nc_luma[mb_idx * 16 + blk] = tc;
                                        for (p, v) in coeffs {
                                            luma_ac[blk][p] = v;
                                        }
                                    }
                                }
                            }

                            // Reconstruct luma: unzigzag DC, Hadamard, dequant
                            let mut luma_dc_raster = [0i32; 16];
                            for i in 0..16 {
                                let (r, c) = ZIGZAG_4X4[i];
                                luma_dc_raster[r * 4 + c] = luma_dc[i];
                            }
                            inverse_hadamard_4x4(&mut luma_dc_raster);
                            dequant_luma_dc_i16x16(
                                &mut luma_dc_raster,
                                qp_y,
                                pps.scaling_list_4x4[0][0],
                            );

                            const DC_RASTER_TO_BLOCK_P: [usize; 16] =
                                [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15];
                            let mut luma_residual = [0i32; 256];
                            for blk in 0..16 {
                                let (blk_row, blk_col) = BLOCK_INDEX_TO_OFFSET[blk];
                                let mut block_raster = [0i32; 16];
                                let dc_idx =
                                    DC_RASTER_TO_BLOCK_P.iter().position(|&b| b == blk).unwrap();
                                block_raster[0] = luma_dc_raster[dc_idx];

                                if cbp_luma != 0 {
                                    for scan_idx in 0..15 {
                                        let (r, c) = ZIGZAG_4X4[scan_idx + 1];
                                        block_raster[r * 4 + c] = luma_ac[blk][scan_idx];
                                    }
                                    dequant_4x4_ac_raster(
                                        &mut block_raster,
                                        qp_y,
                                        &pps.scaling_list_4x4[0],
                                    );
                                }
                                inverse_dct_4x4(&mut block_raster);

                                for r in 0..4 {
                                    for c in 0..4 {
                                        luma_residual[(blk_row + r) * 16 + blk_col + c] =
                                            block_raster[r * 4 + c];
                                    }
                                }
                            }

                            // I16x16 prediction + residual
                            make_ctx!().reconstruct_luma_16x16(
                                mb_x,
                                mb_y,
                                i16_pred,
                                &luma_residual,
                                above_mb_avail,
                                left_mb_avail,
                                above_left_mb_avail,
                            );

                            // Chroma prediction
                            let chroma_width = (width / 2) as usize;
                            let chroma_mb_x = mb_x / 2;
                            let chroma_mb_y = mb_y / 2;
                            let (pred_u, pred_v) = make_ctx!().predict_chroma_intra(
                                mb_x,
                                mb_y,
                                intra_chroma_pred_mode,
                                above_mb_avail,
                                left_mb_avail,
                                above_left_mb_avail,
                            );

                            // Chroma DC residual
                            let mut chroma_dc_cb = [0i32; 4];
                            let mut chroma_dc_cr = [0i32; 4];
                            if cbp_chroma >= 1 {
                                let left_dc_nz = if !mb_idx.is_multiple_of(mb_width as usize)
                                    && mb_slice_id[mb_idx - 1] == this_slice_id
                                {
                                    (mb_cbp[mb_idx - 1] >> 6) & 1 != 0
                                } else {
                                    true
                                };
                                let top_dc_nz = if mb_idx >= mb_width as usize
                                    && mb_slice_id[mb_idx - mb_width as usize] == this_slice_id
                                {
                                    (mb_cbp[mb_idx - mb_width as usize] >> 6) & 1 != 0
                                } else {
                                    true
                                };
                                if cr.decode_coded_block_flag(st, 3, left_dc_nz, top_dc_nz) {
                                    let (coeffs, _) = cr.decode_residual_cabac(st, 3, 4);
                                    for (p, v) in coeffs {
                                        chroma_dc_cb[p] = v;
                                    }
                                    mb_cbp[mb_idx] |= 0x40;
                                }
                                let left_dc_nz_cr = if !mb_idx.is_multiple_of(mb_width as usize)
                                    && mb_slice_id[mb_idx - 1] == this_slice_id
                                {
                                    (mb_cbp[mb_idx - 1] >> 7) & 1 != 0
                                } else {
                                    true
                                };
                                let top_dc_nz_cr = if mb_idx >= mb_width as usize
                                    && mb_slice_id[mb_idx - mb_width as usize] == this_slice_id
                                {
                                    (mb_cbp[mb_idx - mb_width as usize] >> 7) & 1 != 0
                                } else {
                                    true
                                };
                                if cr.decode_coded_block_flag(st, 3, left_dc_nz_cr, top_dc_nz_cr) {
                                    let (coeffs, _) = cr.decode_residual_cabac(st, 3, 4);
                                    for (p, v) in coeffs {
                                        chroma_dc_cr[p] = v;
                                    }
                                    mb_cbp[mb_idx] |= 0x80;
                                }
                            }

                            // Chroma AC residual
                            let mut chroma_ac_cb = [[0i32; 15]; 4];
                            let mut chroma_ac_cr = [[0i32; 15]; 4];
                            if cbp_chroma >= 2 {
                                for blk in 0..4 {
                                    let left_nz = cabac_neighbor_nz_chroma(
                                        &nc_cb,
                                        mb_idx,
                                        mb_width as usize,
                                        blk,
                                        true,
                                        true,
                                        &mb_slice_id,
                                        this_slice_id,
                                    );
                                    let top_nz = cabac_neighbor_nz_chroma(
                                        &nc_cb,
                                        mb_idx,
                                        mb_width as usize,
                                        blk,
                                        false,
                                        true,
                                        &mb_slice_id,
                                        this_slice_id,
                                    );
                                    if cr.decode_coded_block_flag(st, 4, left_nz, top_nz) {
                                        let (coeffs, tc) = cr.decode_residual_cabac(st, 4, 15);
                                        nc_cb[mb_idx * 4 + blk] = tc;
                                        for (p, v) in coeffs {
                                            chroma_ac_cb[blk][p] = v;
                                        }
                                    }
                                }
                                for blk in 0..4 {
                                    let left_nz = cabac_neighbor_nz_chroma(
                                        &nc_cr,
                                        mb_idx,
                                        mb_width as usize,
                                        blk,
                                        true,
                                        true,
                                        &mb_slice_id,
                                        this_slice_id,
                                    );
                                    let top_nz = cabac_neighbor_nz_chroma(
                                        &nc_cr,
                                        mb_idx,
                                        mb_width as usize,
                                        blk,
                                        false,
                                        true,
                                        &mb_slice_id,
                                        this_slice_id,
                                    );
                                    if cr.decode_coded_block_flag(st, 4, left_nz, top_nz) {
                                        let (coeffs, tc) = cr.decode_residual_cabac(st, 4, 15);
                                        nc_cr[mb_idx * 4 + blk] = tc;
                                        for (p, v) in coeffs {
                                            chroma_ac_cr[blk][p] = v;
                                        }
                                    }
                                }
                            }
                            mb_cbp[mb_idx] |= (cbp_luma as u16) | ((cbp_chroma as u16) << 4);

                            // Chroma reconstruction
                            for (plane_dc, plane_ac, pred_plane, frame_plane, scale_idx) in [
                                (
                                    &mut chroma_dc_cb,
                                    &chroma_ac_cb,
                                    &pred_u,
                                    &mut frame.u,
                                    1usize,
                                ),
                                (
                                    &mut chroma_dc_cr,
                                    &chroma_ac_cr,
                                    &pred_v,
                                    &mut frame.v,
                                    2usize,
                                ),
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
                                        dequant_4x4_ac_raster(
                                            &mut block_raster,
                                            qp_c,
                                            chroma_scale,
                                        );
                                    }
                                    inverse_dct_4x4(&mut block_raster);
                                    for r in 0..4 {
                                        for c in 0..4 {
                                            chroma_residual[(blk_row + r) * 8 + blk_col + c] =
                                                block_raster[r * 4 + c];
                                        }
                                    }
                                }
                                for y in 0..8 {
                                    for x in 0..8 {
                                        let val = (pred_plane[y * 8 + x] as i32
                                            + chroma_residual[y * 8 + x])
                                            .clamp(0, 255)
                                            as u8;
                                        frame_plane
                                            [(chroma_mb_y + y) * chroma_width + chroma_mb_x + x] =
                                            val;
                                    }
                                }
                            }

                            mb_info[mb_idx] = MbInfo {
                                mb_type: MbType::Intra,
                                qp_y,
                                ..Default::default()
                            };
                        } else {
                            // I_PCM in P/B via CABAC
                            let pcm_pos = cr.pcm_byte_position();
                            let pcm_data = &nal.rbsp[pcm_pos..];
                            let mut off = 0;
                            for r in 0..16 {
                                for c in 0..16 {
                                    frame.y[(mb_y + r) * stride + mb_x + c] = pcm_data[off];
                                    off += 1;
                                }
                            }
                            let cw = (width / 2) as usize;
                            let cx = mb_x / 2;
                            let cy = mb_y / 2;
                            for r in 0..8 {
                                for c in 0..8 {
                                    frame.u[(cy + r) * cw + cx + c] = pcm_data[off];
                                    off += 1;
                                }
                            }
                            for r in 0..8 {
                                for c in 0..8 {
                                    frame.v[(cy + r) * cw + cx + c] = pcm_data[off];
                                    off += 1;
                                }
                            }
                            for blk in 0..16 {
                                nc_luma[mb_idx * 16 + blk] = 16;
                            }
                            for blk in 0..4 {
                                nc_cb[mb_idx * 4 + blk] = 16;
                                nc_cr[mb_idx * 4 + blk] = 16;
                            }
                            cr.reinit(pcm_pos + off);
                            prev_mb_qp = 0;
                            last_qp_delta_nonzero = false;
                            mb_info[mb_idx] = MbInfo {
                                mb_type: MbType::Ipcm,
                                qp_y: 0,
                                ..Default::default()
                            };
                        }
                        mb_idx += 1;
                        continue;
                    }

                    // Inter MB: decode using CABAC syntax elements
                    if is_p_slice {
                        // P-slice inter: types 0-4
                        let is_p8x8 = raw_mb_type == 3 || raw_mb_type == 4;
                        let (part_w, part_h, num_parts) = match raw_mb_type {
                            0 => (16usize, 16usize, 1usize), // P_L0_16x16
                            1 => (16, 8, 2),                 // P_L0_L0_16x8
                            2 => (8, 16, 2),                 // P_L0_L0_8x16
                            3 | 4 => (8, 8, 4),              // P_8x8 / P_8x8ref0
                            _ => unreachable!(),
                        };

                        let sub_mb_origins = [(0usize, 0usize), (0, 8), (8, 0), (8, 8)];
                        let mut sub_mb_types = [0u32; 4];
                        if is_p8x8 {
                            // Sub-MB types
                            for smt in &mut sub_mb_types {
                                *smt = cr.decode_p_sub_mb_type(st);
                            }
                            // Parse ref_idx
                            let mut sub_ref = [0i8; 4];
                            if raw_mb_type == 3 {
                                for (smb, sr) in sub_ref.iter_mut().enumerate() {
                                    if header.num_ref_idx_l0_active > 1 {
                                        let (sy, sx) = sub_mb_origins[smb];
                                        let (left_ref, top_ref) = cabac_neighbor_ref(
                                            &ref_idx_store_l0,
                                            mb_idx,
                                            mb_width as usize,
                                            sy,
                                            sx,
                                            &mb_slice_id,
                                            this_slice_id,
                                        );
                                        *sr = cr.decode_ref_idx(st, left_ref, top_ref);
                                    }
                                    // Write ref_idx immediately for neighbor context
                                    let (sy, sx) = sub_mb_origins[smb];
                                    for r in (0..8).step_by(4) {
                                        for c in (0..8).step_by(4) {
                                            let lr = (sy + r) / 4;
                                            let lc = (sx + c) / 4;
                                            if let Some(blk) = BLOCK_INDEX_TO_OFFSET
                                                .iter()
                                                .position(|&(br, bc)| br / 4 == lr && bc / 4 == lc)
                                            {
                                                ref_idx_store_l0[mb_idx * 16 + blk] = *sr;
                                            }
                                        }
                                    }
                                }
                            }
                            // Parse MVDs and reconstruct
                            for smb in 0..4 {
                                let (sy, sx) = sub_mb_origins[smb];
                                let ref_idx = sub_ref[smb];
                                let sub_parts_layout: Vec<(usize, usize, usize, usize)> =
                                    match sub_mb_types[smb] {
                                        0 => vec![(0, 0, 8, 8)],
                                        1 => vec![(0, 0, 8, 4), (0, 4, 8, 4)],
                                        2 => vec![(0, 0, 4, 8), (4, 0, 4, 8)],
                                        3 => vec![
                                            (0, 0, 4, 4),
                                            (4, 0, 4, 4),
                                            (0, 4, 4, 4),
                                            (4, 4, 4, 4),
                                        ],
                                        _ => return Err(DecodeError::from("invalid sub_mb_type")),
                                    };
                                for &(dx, dy, spw, sph) in &sub_parts_layout {
                                    let px = sx + dx;
                                    let py = sy + dy;
                                    let amvd_x = cabac_amvd(
                                        &mvd_store,
                                        mb_idx,
                                        mb_width as usize,
                                        py,
                                        px,
                                        0,
                                        &mb_slice_id,
                                        this_slice_id,
                                    );
                                    let amvd_y = cabac_amvd(
                                        &mvd_store,
                                        mb_idx,
                                        mb_width as usize,
                                        py,
                                        px,
                                        1,
                                        &mb_slice_id,
                                        this_slice_id,
                                    );
                                    let mvd_x = cr.decode_mvd_comp(st, 40, amvd_x) as i16;
                                    let mvd_y = cr.decode_mvd_comp(st, 47, amvd_y) as i16;
                                    let (mvp_x, mvp_y) = predict_mv_sub(
                                        &mv_store_l0,
                                        &ref_idx_store_l0,
                                        mb_idx,
                                        mb_width as usize,
                                        px,
                                        py,
                                        spw,
                                        sph,
                                        ref_idx,
                                        &mb_slice_id,
                                        this_slice_id,
                                    );
                                    let mv = [mvp_x + mvd_x, mvp_y + mvd_y];
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
                                                mvd_store[mb_idx * 16 + blk] = [mvd_x, mvd_y];
                                            }
                                        }
                                    }
                                    // MC
                                    let ref_pic = &ref_pic_list
                                        [(ref_idx as usize).min(ref_pic_list.len() - 1)];
                                    let mut luma_pred = vec![0u8; spw * sph];
                                    inter_pred::luma_mc(
                                        ref_pic,
                                        (mb_x + px) as i32,
                                        (mb_y + py) as i32,
                                        mv[0] as i32,
                                        mv[1] as i32,
                                        spw,
                                        sph,
                                        &mut luma_pred,
                                    );
                                    if use_weight == 1 {
                                        wctx.apply_uni(
                                            &mut luma_pred,
                                            0,
                                            ref_idx as usize,
                                            false,
                                            0,
                                        );
                                    }
                                    for r in 0..sph {
                                        for c in 0..spw {
                                            frame.y[(mb_y + py + r) * stride + mb_x + px + c] =
                                                luma_pred[r * spw + c];
                                        }
                                    }
                                }
                            }
                            // Chroma MC for P_8x8 (per sub-partition)
                            let cw = (width / 2) as usize;
                            let cx = mb_x / 2;
                            let cy = mb_y / 2;
                            for smb in 0..4 {
                                let (sy, sx) = sub_mb_origins[smb];
                                let ref_pic = &ref_pic_list
                                    [(sub_ref[smb] as usize).min(ref_pic_list.len() - 1)];
                                let sub_parts: Vec<(usize, usize, usize, usize)> =
                                    match sub_mb_types[smb] {
                                        0 => vec![(0, 0, 8, 8)],
                                        1 => vec![(0, 0, 8, 4), (0, 4, 8, 4)],
                                        2 => vec![(0, 0, 4, 8), (4, 0, 4, 8)],
                                        3 => vec![
                                            (0, 0, 4, 4),
                                            (4, 0, 4, 4),
                                            (0, 4, 4, 4),
                                            (4, 4, 4, 4),
                                        ],
                                        _ => vec![(0, 0, 8, 8)],
                                    };
                                for &(dx, dy, spw, sph) in &sub_parts {
                                    let px = sx + dx;
                                    let py = sy + dy;
                                    let blk_idx = BLOCK_INDEX_TO_OFFSET
                                        .iter()
                                        .position(|&(br, bc)| br == py && bc == px)
                                        .unwrap_or(0);
                                    let mv = mv_store_l0[mb_idx * 16 + blk_idx];
                                    let ccx = px / 2;
                                    let ccy = py / 2;
                                    let ccw = spw.max(2) / 2;
                                    let cch = sph.max(2) / 2;
                                    if ccw == 0 || cch == 0 {
                                        continue;
                                    }
                                    let mut cb_pred = vec![0u8; ccw * cch];
                                    let mut cr_pred_buf = vec![0u8; ccw * cch];
                                    inter_pred::chroma_mc(
                                        &ref_pic.u,
                                        cw,
                                        (height / 2) as usize,
                                        (cx + ccx) as i32,
                                        (cy + ccy) as i32,
                                        mv[0] as i32,
                                        mv[1] as i32,
                                        ccw,
                                        cch,
                                        &mut cb_pred,
                                    );
                                    inter_pred::chroma_mc(
                                        &ref_pic.v,
                                        cw,
                                        (height / 2) as usize,
                                        (cx + ccx) as i32,
                                        (cy + ccy) as i32,
                                        mv[0] as i32,
                                        mv[1] as i32,
                                        ccw,
                                        cch,
                                        &mut cr_pred_buf,
                                    );
                                    if use_weight == 1 {
                                        wctx.apply_uni(
                                            &mut cb_pred,
                                            0,
                                            sub_ref[smb] as usize,
                                            true,
                                            0,
                                        );
                                        wctx.apply_uni(
                                            &mut cr_pred_buf,
                                            0,
                                            sub_ref[smb] as usize,
                                            true,
                                            1,
                                        );
                                    }
                                    for r in 0..cch {
                                        for c in 0..ccw {
                                            frame.u[(cy + ccy + r) * cw + cx + ccx + c] =
                                                cb_pred[r * ccw + c];
                                            frame.v[(cy + ccy + r) * cw + cx + ccx + c] =
                                                cr_pred_buf[r * ccw + c];
                                        }
                                    }
                                }
                            }
                        } else {
                            // P_L0_16x16, P16x8, P8x16
                            let mut part_ref = [0i8; 2];
                            for (p, ref_entry) in part_ref.iter_mut().enumerate().take(num_parts) {
                                if header.num_ref_idx_l0_active > 1 {
                                    let (py_off, px_off) = match raw_mb_type {
                                        1 => (p * 8, 0),
                                        2 => (0, p * 8),
                                        _ => (0, 0),
                                    };
                                    let (left_ref, top_ref) = cabac_neighbor_ref(
                                        &ref_idx_store_l0,
                                        mb_idx,
                                        mb_width as usize,
                                        py_off,
                                        px_off,
                                        &mb_slice_id,
                                        this_slice_id,
                                    );
                                    *ref_entry = cr.decode_ref_idx(st, left_ref, top_ref);
                                }
                                // Write ref_idx to store immediately so the next
                                // partition can see it as a neighbor.
                                let (py_off, px_off) = match raw_mb_type {
                                    1 => (p * 8, 0),
                                    2 => (0, p * 8),
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
                                            ref_idx_store_l0[mb_idx * 16 + blk] = *ref_entry;
                                        }
                                    }
                                }
                            }
                            for p in 0..num_parts {
                                let (py_off, px_off) = match raw_mb_type {
                                    1 => (p * 8, 0),
                                    2 => (0, p * 8),
                                    _ => (0, 0),
                                };
                                let amvd_x = cabac_amvd(
                                    &mvd_store,
                                    mb_idx,
                                    mb_width as usize,
                                    py_off,
                                    px_off,
                                    0,
                                    &mb_slice_id,
                                    this_slice_id,
                                );
                                let amvd_y = cabac_amvd(
                                    &mvd_store,
                                    mb_idx,
                                    mb_width as usize,
                                    py_off,
                                    px_off,
                                    1,
                                    &mb_slice_id,
                                    this_slice_id,
                                );
                                let mvd_x = cr.decode_mvd_comp(st, 40, amvd_x) as i16;
                                let mvd_y = cr.decode_mvd_comp(st, 47, amvd_y) as i16;
                                let (mvp_x, mvp_y) = predict_mv(
                                    &mv_store_l0,
                                    &ref_idx_store_l0,
                                    mb_idx,
                                    mb_width as usize,
                                    p,
                                    part_w,
                                    part_h,
                                    part_ref[p],
                                    &mb_slice_id,
                                    this_slice_id,
                                );
                                let mv = [mvp_x + mvd_x, mvp_y + mvd_y];
                                // Store MV, ref, and MVD
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
                                            mvd_store[mb_idx * 16 + blk] = [mvd_x, mvd_y];
                                        }
                                    }
                                }

                                // MC
                                let ref_pic = &ref_pic_list
                                    [(part_ref[p] as usize).min(ref_pic_list.len() - 1)];
                                let mut luma_pred = vec![0u8; part_w * part_h];
                                inter_pred::luma_mc(
                                    ref_pic,
                                    (mb_x + px_off) as i32,
                                    (mb_y + py_off) as i32,
                                    mv[0] as i32,
                                    mv[1] as i32,
                                    part_w,
                                    part_h,
                                    &mut luma_pred,
                                );
                                if use_weight == 1 {
                                    wctx.apply_uni(
                                        &mut luma_pred,
                                        0,
                                        part_ref[p] as usize,
                                        false,
                                        0,
                                    );
                                }
                                for r in 0..part_h {
                                    for c in 0..part_w {
                                        frame.y[(mb_y + py_off + r) * stride + mb_x + px_off + c] =
                                            luma_pred[r * part_w + c];
                                    }
                                }

                                // Chroma
                                let cw = (width / 2) as usize;
                                let cx_off = px_off / 2;
                                let cy_off = py_off / 2;
                                let chroma_mb_x = mb_x / 2;
                                let chroma_mb_y = mb_y / 2;
                                let chw = part_w.max(2) / 2;
                                let chh = part_h.max(2) / 2;
                                let mut cb_pred = vec![0u8; chw * chh];
                                let mut cr_pred_buf = vec![0u8; chw * chh];
                                inter_pred::chroma_mc(
                                    &ref_pic.u,
                                    cw,
                                    (height / 2) as usize,
                                    (chroma_mb_x + cx_off) as i32,
                                    (chroma_mb_y + cy_off) as i32,
                                    mv[0] as i32,
                                    mv[1] as i32,
                                    chw,
                                    chh,
                                    &mut cb_pred,
                                );
                                inter_pred::chroma_mc(
                                    &ref_pic.v,
                                    cw,
                                    (height / 2) as usize,
                                    (chroma_mb_x + cx_off) as i32,
                                    (chroma_mb_y + cy_off) as i32,
                                    mv[0] as i32,
                                    mv[1] as i32,
                                    chw,
                                    chh,
                                    &mut cr_pred_buf,
                                );
                                if use_weight == 1 {
                                    wctx.apply_uni(&mut cb_pred, 0, part_ref[p] as usize, true, 0);
                                    wctx.apply_uni(
                                        &mut cr_pred_buf,
                                        0,
                                        part_ref[p] as usize,
                                        true,
                                        1,
                                    );
                                }
                                for r in 0..chh {
                                    for c in 0..chw {
                                        frame.u[(chroma_mb_y + cy_off + r) * cw
                                            + chroma_mb_x
                                            + cx_off
                                            + c] = cb_pred[r * chw + c];
                                        frame.v[(chroma_mb_y + cy_off + r) * cw
                                            + chroma_mb_x
                                            + cx_off
                                            + c] = cr_pred_buf[r * chw + c];
                                    }
                                }
                            }
                        }

                        // Residual (inter uses CBP_INTER_TABLE equivalent from CABAC)
                        let left_cbp_raw = if !mb_idx.is_multiple_of(mb_width as usize)
                            && mb_slice_id[mb_idx - 1] == this_slice_id
                        {
                            mb_cbp[mb_idx - 1]
                        } else {
                            0x00Fu16
                        };
                        let top_cbp_raw = if mb_idx >= mb_width as usize
                            && mb_slice_id[mb_idx - mb_width as usize] == this_slice_id
                        {
                            mb_cbp[mb_idx - mb_width as usize]
                        } else {
                            0x00Fu16
                        };
                        let left_cbp = ((left_cbp_raw & 0x7F0)
                            | (left_cbp_raw & 2)
                            | (((left_cbp_raw >> 2) & 2) << 2))
                            as u8;
                        let top_cbp = top_cbp_raw as u8;
                        let cbp_luma = cr.decode_cbp_luma(st, left_cbp, top_cbp);
                        let left_cbp_c = ((left_cbp_raw >> 4) & 3) as u8;
                        let top_cbp_c = ((top_cbp_raw >> 4) & 3) as u8;
                        let cbp_chroma = cr.decode_cbp_chroma(st, left_cbp_c, top_cbp_c);
                        mb_cbp[mb_idx] = (cbp_luma as u16) | ((cbp_chroma as u16) << 4);
                        // 8x8 transform flag for inter MBs (spec 7.3.5).
                        // Only present when noSubMbPartSizeLessThan8x8Flag is true.
                        // For P_8x8, this requires all sub_mb_types to be 0 (8x8 sub-partitions).
                        let no_sub_less_than_8x8 = if is_p8x8 {
                            sub_mb_types.iter().all(|&smt| smt == 0)
                        } else {
                            true // P_16x16, P_16x8, P_8x16 always have partitions >= 8x8
                        };
                        let use_8x8_inter =
                            if pps.transform_8x8_mode_flag && cbp_luma != 0 && no_sub_less_than_8x8
                            {
                                let nts = {
                                    let left = if !mb_idx.is_multiple_of(mb_width as usize)
                                        && mb_slice_id[mb_idx - 1] == this_slice_id
                                    {
                                        mb_is_8x8dct[mb_idx - 1] as usize
                                    } else {
                                        0
                                    };
                                    let top = if mb_idx >= mb_width as usize
                                        && mb_slice_id[mb_idx - mb_width as usize] == this_slice_id
                                    {
                                        mb_is_8x8dct[mb_idx - mb_width as usize] as usize
                                    } else {
                                        0
                                    };
                                    left + top
                                };
                                cr.get_cabac(&mut st[399 + nts]) != 0
                            } else {
                                false
                            };
                        mb_is_8x8dct[mb_idx] = use_8x8_inter;

                        let qp_y = if cbp_luma != 0 || cbp_chroma != 0 {
                            let delta = cr.decode_mb_qp_delta(st, last_qp_delta_nonzero);
                            last_qp_delta_nonzero = delta != 0;
                            ((prev_mb_qp + delta + 52) % 52 + 52) % 52
                        } else {
                            last_qp_delta_nonzero = false;
                            prev_mb_qp
                        };
                        prev_mb_qp = qp_y;
                        // Decode and add luma residual
                        if cbp_luma != 0 {
                            let mut luma_residual = [0i32; 256];
                            if use_8x8_inter {
                                // 8x8 transform: 4 blocks of 64 coefficients
                                let scale_8x8 = &pps.scaling_list_8x8[1]; // inter
                                for i8x8 in 0..4 {
                                    if cbp_luma & (1 << i8x8) == 0 {
                                        continue;
                                    }
                                    // Note: cat=5 (8x8 luma) does NOT use coded_block_flag.
                                    // The CBP luma bit alone indicates coefficients are present.
                                    let (coeffs, tc) = cr.decode_residual_cabac(st, 5, 64);
                                    let tc_per = tc.div_ceil(4);
                                    for sub in 0..4 {
                                        nc_luma[mb_idx * 16 + i8x8 * 4 + sub] = tc_per;
                                    }
                                    let mut block_8x8 = [0i32; 64];
                                    for (pos, val) in &coeffs {
                                        block_8x8[ZIGZAG_8X8_CABAC[*pos]] = *val;
                                    }
                                    dequant_8x8(&mut block_8x8, qp_y, scale_8x8);
                                    inverse_dct_8x8(&mut block_8x8);
                                    let row_off = (i8x8 / 2) * 8;
                                    let col_off = (i8x8 % 2) * 8;
                                    for r in 0..8 {
                                        for c in 0..8 {
                                            luma_residual[(row_off + r) * 16 + col_off + c] =
                                                block_8x8[r * 8 + c];
                                        }
                                    }
                                }
                            } else {
                                // 4x4 transform (existing path)
                                for blk in 0..16 {
                                    if cbp_luma & (1 << (blk / 4)) != 0 {
                                        let left_nz = cabac_neighbor_nz_luma(
                                            &nc_luma,
                                            mb_idx,
                                            mb_width as usize,
                                            blk,
                                            true,
                                            false,
                                            &mb_slice_id,
                                            this_slice_id,
                                        );
                                        let top_nz = cabac_neighbor_nz_luma(
                                            &nc_luma,
                                            mb_idx,
                                            mb_width as usize,
                                            blk,
                                            false,
                                            false,
                                            &mb_slice_id,
                                            this_slice_id,
                                        );
                                        let cbf =
                                            cr.decode_coded_block_flag(st, 2, left_nz, top_nz);
                                        if cbf {
                                            let (coeffs, tc) = cr.decode_residual_cabac(st, 2, 16);
                                            nc_luma[mb_idx * 16 + blk] = tc;
                                            let mut block_coeffs = [0i32; 16];
                                            for (pos, val) in &coeffs {
                                                let (r, c) = ZIGZAG_4X4[*pos];
                                                block_coeffs[r * 4 + c] = *val;
                                            }
                                            dequant_4x4_full(
                                                &mut block_coeffs,
                                                qp_y,
                                                &pps.scaling_list_4x4[3],
                                            );
                                            inverse_dct_4x4(&mut block_coeffs);
                                            let (blk_row, blk_col) = BLOCK_INDEX_TO_OFFSET[blk];
                                            for r in 0..4 {
                                                for c in 0..4 {
                                                    luma_residual
                                                        [(blk_row + r) * 16 + blk_col + c] =
                                                        block_coeffs[r * 4 + c];
                                                }
                                            }
                                        }
                                    }
                                }
                            } // close if use_8x8_inter else
                              // Add residual to prediction
                            for r in 0..16 {
                                for c in 0..16 {
                                    let val = (frame.y[(mb_y + r) * stride + mb_x + c] as i32
                                        + luma_residual[r * 16 + c])
                                        .clamp(0, 255)
                                        as u8;
                                    frame.y[(mb_y + r) * stride + mb_x + c] = val;
                                }
                            }
                        }

                        // Chroma residual
                        let qp_c = chroma_qp(qp_y, pps.chroma_qp_index_offset);
                        if cbp_chroma >= 1 {
                            let mut chroma_dc_cb = [0i32; 4];
                            let mut chroma_dc_cr = [0i32; 4];
                            let left_dc_nz = if !mb_idx.is_multiple_of(mb_width as usize)
                                && mb_slice_id[mb_idx - 1] == this_slice_id
                            {
                                (mb_cbp[mb_idx - 1] >> 6) & 1 != 0
                            } else {
                                false
                            };
                            let top_dc_nz = if mb_idx >= mb_width as usize
                                && mb_slice_id[mb_idx - mb_width as usize] == this_slice_id
                            {
                                (mb_cbp[mb_idx - mb_width as usize] >> 6) & 1 != 0
                            } else {
                                false
                            };
                            if cr.decode_coded_block_flag(st, 3, left_dc_nz, top_dc_nz) {
                                let (coeffs, _) = cr.decode_residual_cabac(st, 3, 4);
                                for (pos, val) in coeffs {
                                    chroma_dc_cb[pos] = val;
                                }
                                mb_cbp[mb_idx] |= 0x40;
                            }
                            let left_dc_cr = if !mb_idx.is_multiple_of(mb_width as usize)
                                && mb_slice_id[mb_idx - 1] == this_slice_id
                            {
                                (mb_cbp[mb_idx - 1] >> 7) & 1 != 0
                            } else {
                                false
                            };
                            let top_dc_cr = if mb_idx >= mb_width as usize
                                && mb_slice_id[mb_idx - mb_width as usize] == this_slice_id
                            {
                                (mb_cbp[mb_idx - mb_width as usize] >> 7) & 1 != 0
                            } else {
                                false
                            };
                            if cr.decode_coded_block_flag(st, 3, left_dc_cr, top_dc_cr) {
                                let (coeffs, _) = cr.decode_residual_cabac(st, 3, 4);
                                for (pos, val) in coeffs {
                                    chroma_dc_cr[pos] = val;
                                }
                                mb_cbp[mb_idx] |= 0x80;
                            }

                            let chroma_width = (width / 2) as usize;
                            let chroma_mb_x = mb_x / 2;
                            let chroma_mb_y = mb_y / 2;
                            for (plane_dc, frame_plane, scale_idx) in [
                                (&mut chroma_dc_cb, &mut frame.u, 4usize),
                                (&mut chroma_dc_cr, &mut frame.v, 5usize),
                            ] {
                                let chroma_scale = &pps.scaling_list_4x4[scale_idx];
                                inverse_hadamard_2x2(plane_dc);
                                dequant_chroma_dc(plane_dc, qp_c, chroma_scale[0]);
                                let mut chroma_residual = [0i32; 64];
                                for blk in 0..4 {
                                    let blk_row = (blk / 2) * 4;
                                    let blk_col = (blk % 2) * 4;
                                    let mut block_raster = [0i32; 16];
                                    block_raster[0] = plane_dc[blk];
                                    if cbp_chroma >= 2 {
                                        let nc_arr = if scale_idx == 4 { &nc_cb } else { &nc_cr };
                                        let left_nz = cabac_neighbor_nz_chroma(
                                            nc_arr,
                                            mb_idx,
                                            mb_width as usize,
                                            blk,
                                            true,
                                            false,
                                            &mb_slice_id,
                                            this_slice_id,
                                        );
                                        let top_nz = cabac_neighbor_nz_chroma(
                                            nc_arr,
                                            mb_idx,
                                            mb_width as usize,
                                            blk,
                                            false,
                                            false,
                                            &mb_slice_id,
                                            this_slice_id,
                                        );
                                        if cr.decode_coded_block_flag(st, 4, left_nz, top_nz) {
                                            let (coeffs, tc) = cr.decode_residual_cabac(st, 4, 15);
                                            if scale_idx == 4 {
                                                nc_cb[mb_idx * 4 + blk] = tc;
                                            } else {
                                                nc_cr[mb_idx * 4 + blk] = tc;
                                            }
                                            for (pos, val) in coeffs {
                                                let (r, c) = ZIGZAG_4X4[pos + 1];
                                                block_raster[r * 4 + c] = val;
                                            }
                                            dequant_4x4_ac_raster(
                                                &mut block_raster,
                                                qp_c,
                                                chroma_scale,
                                            );
                                        }
                                    }
                                    inverse_dct_4x4(&mut block_raster);
                                    for r in 0..4 {
                                        for c in 0..4 {
                                            chroma_residual[(blk_row + r) * 8 + blk_col + c] =
                                                block_raster[r * 4 + c];
                                        }
                                    }
                                }
                                for y in 0..8 {
                                    for x in 0..8 {
                                        let val = (frame_plane
                                            [(chroma_mb_y + y) * chroma_width + chroma_mb_x + x]
                                            as i32
                                            + chroma_residual[y * 8 + x])
                                            .clamp(0, 255)
                                            as u8;
                                        frame_plane
                                            [(chroma_mb_y + y) * chroma_width + chroma_mb_x + x] =
                                            val;
                                    }
                                }
                            }
                        }

                        mb_info[mb_idx] = MbInfo {
                            mb_type: MbType::Inter,
                            qp_y,
                            ..Default::default()
                        };
                        mb_idx += 1;
                        continue;
                    } else {
                        // B-slice inter with CABAC
                        // raw_mb_type mapping (from decode_b_mb_type):
                        // 0: B_Direct_16x16
                        // 1: B_L0_16x16, 2: B_L1_16x16, 3: B_Bi_16x16
                        // 4-21: B 16x8/8x16 partition variants
                        // 22: B_8x8
                        // Track whether all sub-partitions are >= 8x8 (for transform_size_8x8_flag)
                        let mut no_sub_less_than_8x8_b = true;

                        #[rustfmt::skip]
                        #[allow(clippy::type_complexity)]
                        const B_PART_TABLE: [(usize, usize, [(bool, bool); 2]); 18] = [
                            (16, 8, [(true,false),(true,false)]),  // 4: B_L0_L0_16x8
                            (8, 16, [(true,false),(true,false)]),  // 5: B_L0_L0_8x16
                            (16, 8, [(false,true),(false,true)]),  // 6: B_L1_L1_16x8
                            (8, 16, [(false,true),(false,true)]),  // 7: B_L1_L1_8x16
                            (16, 8, [(true,false),(false,true)]),  // 8: B_L0_L1_16x8
                            (8, 16, [(true,false),(false,true)]),  // 9: B_L0_L1_8x16
                            (16, 8, [(false,true),(true,false)]),  // 10: B_L1_L0_16x8
                            (8, 16, [(false,true),(true,false)]),  // 11: B_L1_L0_8x16
                            (16, 8, [(true,false),(true,true)]),   // 12: B_L0_Bi_16x8
                            (8, 16, [(true,false),(true,true)]),   // 13: B_L0_Bi_8x16
                            (16, 8, [(false,true),(true,true)]),   // 14: B_L1_Bi_16x8
                            (8, 16, [(false,true),(true,true)]),   // 15: B_L1_Bi_8x16
                            (16, 8, [(true,true),(true,false)]),   // 16: B_Bi_L0_16x8
                            (8, 16, [(true,true),(true,false)]),   // 17: B_Bi_L0_8x16
                            (16, 8, [(true,true),(false,true)]),   // 18: B_Bi_L1_16x8
                            (8, 16, [(true,true),(false,true)]),   // 19: B_Bi_L1_8x16
                            (16, 8, [(true,true),(true,true)]),    // 20: B_Bi_Bi_16x8
                            (8, 16, [(true,true),(true,true)]),    // 21: B_Bi_Bi_8x16
                        ];

                        #[rustfmt::skip]
                        const B_SUB_TABLE: [(usize, usize, bool, bool); 13] = [
                            (8, 8, false, false), // 0: B_Direct_8x8
                            (8, 8, true, false),  // 1: B_L0_8x8
                            (8, 8, false, true),  // 2: B_L1_8x8
                            (8, 8, true, true),   // 3: B_Bi_8x8
                            (8, 4, true, false),  // 4: B_L0_8x4
                            (4, 8, true, false),  // 5: B_L0_4x8
                            (8, 4, false, true),  // 6: B_L1_8x4
                            (4, 8, false, true),  // 7: B_L1_4x8
                            (8, 4, true, true),   // 8: B_Bi_8x4
                            (4, 8, true, true),   // 9: B_Bi_4x8
                            (4, 4, true, false),  // 10: B_L0_4x4
                            (4, 4, false, true),  // 11: B_L1_4x4
                            (4, 4, true, true),   // 12: B_Bi_4x4
                        ];

                        struct BSubPart {
                            x: usize,
                            y: usize,
                            w: usize,
                            h: usize,
                            ref_idx_l0: i8,
                            ref_idx_l1: i8,
                            mv_l0: [i16; 2],
                            mv_l1: [i16; 2],
                            pred_l0: bool,
                            pred_l1: bool,
                        }
                        let mut b_sub_parts: Vec<BSubPart> = Vec::new();

                        if raw_mb_type == 0 {
                            // B_Direct_16x16: derive MVs per 4x4 block
                            mb_is_direct[mb_idx] = true;
                            if !sps.direct_8x8_inference_flag {
                                no_sub_less_than_8x8_b = false;
                            }
                            make_ctx!().derive_direct_mvs(
                                mb_idx,
                                0,
                                16,
                                header.direct_spatial_mv_pred_flag,
                                sps.direct_8x8_inference_flag,
                                current_poc,
                                &_ref_pic_list_l0,
                                &_ref_pic_list_l1,
                            );
                            // Build sub_parts, coalescing per-8x8 when MVs are uniform
                            let base = mb_idx * 16;
                            for i8x8 in 0..4 {
                                let blk0 = i8x8 * 4;
                                let mv0 = mv_store_l0[base + blk0];
                                let mv1 = mv_store_l1[base + blk0];
                                let r0 = ref_idx_store_l0[base + blk0];
                                let r1 = ref_idx_store_l1[base + blk0];
                                let uniform = (1..4).all(|sub| {
                                    let b = blk0 + sub;
                                    mv_store_l0[base + b] == mv0
                                        && mv_store_l1[base + b] == mv1
                                        && ref_idx_store_l0[base + b] == r0
                                        && ref_idx_store_l1[base + b] == r1
                                });
                                let (blk_row, blk_col) = BLOCK_INDEX_TO_OFFSET[blk0];
                                if uniform {
                                    b_sub_parts.push(BSubPart {
                                        x: blk_col,
                                        y: blk_row,
                                        w: 8,
                                        h: 8,
                                        ref_idx_l0: r0,
                                        ref_idx_l1: r1,
                                        mv_l0: mv0,
                                        mv_l1: mv1,
                                        pred_l0: r0 >= 0,
                                        pred_l1: r1 >= 0,
                                    });
                                } else {
                                    for sub in 0..4 {
                                        let b = blk0 + sub;
                                        let (br, bc) = BLOCK_INDEX_TO_OFFSET[b];
                                        let rl0 = ref_idx_store_l0[base + b];
                                        let rl1 = ref_idx_store_l1[base + b];
                                        b_sub_parts.push(BSubPart {
                                            x: bc,
                                            y: br,
                                            w: 4,
                                            h: 4,
                                            ref_idx_l0: rl0,
                                            ref_idx_l1: rl1,
                                            mv_l0: mv_store_l0[base + b],
                                            mv_l1: mv_store_l1[base + b],
                                            pred_l0: rl0 >= 0,
                                            pred_l1: rl1 >= 0,
                                        });
                                    }
                                }
                            }
                        } else if raw_mb_type <= 3 {
                            // B_L0_16x16 (1), B_L1_16x16 (2), B_Bi_16x16 (3)
                            let pred_l0 = raw_mb_type == 1 || raw_mb_type == 3;
                            let pred_l1 = raw_mb_type == 2 || raw_mb_type == 3;

                            let mut ref_l0 = 0i8;
                            let mut ref_l1 = 0i8;

                            if pred_l0 && header.num_ref_idx_l0_active > 1 {
                                let (left_ref, top_ref) = cabac_neighbor_ref(
                                    &ref_idx_store_l0,
                                    mb_idx,
                                    mb_width as usize,
                                    0,
                                    0,
                                    &mb_slice_id,
                                    this_slice_id,
                                );
                                ref_l0 = cr.decode_ref_idx(st, left_ref, top_ref);
                            }
                            if pred_l1 && header.num_ref_idx_l1_active > 1 {
                                let (left_ref, top_ref) = cabac_neighbor_ref(
                                    &ref_idx_store_l1,
                                    mb_idx,
                                    mb_width as usize,
                                    0,
                                    0,
                                    &mb_slice_id,
                                    this_slice_id,
                                );
                                ref_l1 = cr.decode_ref_idx(st, left_ref, top_ref);
                            }

                            let mut mv_l0 = [0i16; 2];
                            let mut mv_l1 = [0i16; 2];

                            if pred_l0 {
                                let amvd_x = cabac_amvd(
                                    &mvd_store,
                                    mb_idx,
                                    mb_width as usize,
                                    0,
                                    0,
                                    0,
                                    &mb_slice_id,
                                    this_slice_id,
                                );
                                let amvd_y = cabac_amvd(
                                    &mvd_store,
                                    mb_idx,
                                    mb_width as usize,
                                    0,
                                    0,
                                    1,
                                    &mb_slice_id,
                                    this_slice_id,
                                );
                                let mvd_x = cr.decode_mvd_comp(st, 40, amvd_x) as i16;
                                let mvd_y = cr.decode_mvd_comp(st, 47, amvd_y) as i16;
                                let (mvp_x, mvp_y) = predict_mv(
                                    &mv_store_l0,
                                    &ref_idx_store_l0,
                                    mb_idx,
                                    mb_width as usize,
                                    0,
                                    16,
                                    16,
                                    ref_l0,
                                    &mb_slice_id,
                                    this_slice_id,
                                );
                                mv_l0 = [mvp_x + mvd_x, mvp_y + mvd_y];
                                for blk in 0..16 {
                                    mv_store_l0[mb_idx * 16 + blk] = mv_l0;
                                    ref_idx_store_l0[mb_idx * 16 + blk] = ref_l0;
                                    mvd_store[mb_idx * 16 + blk] = [mvd_x, mvd_y];
                                }
                            }
                            if pred_l1 {
                                let amvd_x = cabac_amvd(
                                    &mvd_store_l1,
                                    mb_idx,
                                    mb_width as usize,
                                    0,
                                    0,
                                    0,
                                    &mb_slice_id,
                                    this_slice_id,
                                );
                                let amvd_y = cabac_amvd(
                                    &mvd_store_l1,
                                    mb_idx,
                                    mb_width as usize,
                                    0,
                                    0,
                                    1,
                                    &mb_slice_id,
                                    this_slice_id,
                                );
                                let mvd_x = cr.decode_mvd_comp(st, 40, amvd_x) as i16;
                                let mvd_y = cr.decode_mvd_comp(st, 47, amvd_y) as i16;
                                let (mvp_x, mvp_y) = predict_mv(
                                    &mv_store_l1,
                                    &ref_idx_store_l1,
                                    mb_idx,
                                    mb_width as usize,
                                    0,
                                    16,
                                    16,
                                    ref_l1,
                                    &mb_slice_id,
                                    this_slice_id,
                                );
                                mv_l1 = [mvp_x + mvd_x, mvp_y + mvd_y];
                                for blk in 0..16 {
                                    mv_store_l1[mb_idx * 16 + blk] = mv_l1;
                                    ref_idx_store_l1[mb_idx * 16 + blk] = ref_l1;
                                    mvd_store_l1[mb_idx * 16 + blk] = [mvd_x, mvd_y];
                                }
                            }

                            // Set default ref_idx for inactive list
                            if !pred_l0 {
                                for blk in 0..16 {
                                    ref_idx_store_l0[mb_idx * 16 + blk] = -1;
                                }
                            }
                            if !pred_l1 {
                                for blk in 0..16 {
                                    ref_idx_store_l1[mb_idx * 16 + blk] = -1;
                                }
                            }

                            b_sub_parts.push(BSubPart {
                                x: 0,
                                y: 0,
                                w: 16,
                                h: 16,
                                ref_idx_l0: if pred_l0 { ref_l0 } else { -1 },
                                ref_idx_l1: if pred_l1 { ref_l1 } else { -1 },
                                mv_l0,
                                mv_l1,
                                pred_l0,
                                pred_l1,
                            });
                        } else if raw_mb_type <= 21 {
                            // B 16x8/8x16 partition variants (mb_type 4-21)
                            let entry = B_PART_TABLE[(raw_mb_type - 4) as usize];
                            let (part_w, part_h) = (entry.0, entry.1);
                            let pred_flags = entry.2;

                            // Parse ref_idx: L0 for all partitions, then L1 for all
                            let mut part_ref_l0 = [-1i8; 2];
                            let mut part_ref_l1 = [-1i8; 2];
                            for p in 0..2 {
                                if pred_flags[p].0 {
                                    if header.num_ref_idx_l0_active > 1 {
                                        let (py_off, px_off) =
                                            if part_h == 8 { (p * 8, 0) } else { (0, p * 8) };
                                        let (left_ref, top_ref) = cabac_neighbor_ref(
                                            &ref_idx_store_l0,
                                            mb_idx,
                                            mb_width as usize,
                                            py_off,
                                            px_off,
                                            &mb_slice_id,
                                            this_slice_id,
                                        );
                                        part_ref_l0[p] = cr.decode_ref_idx(st, left_ref, top_ref);
                                    } else {
                                        part_ref_l0[p] = 0;
                                    }
                                }
                                // Write L0 ref_idx immediately for neighbor context
                                let (py_off, px_off) =
                                    if part_h == 8 { (p * 8, 0) } else { (0, p * 8) };
                                for r in (0..part_h).step_by(4) {
                                    for c in (0..part_w).step_by(4) {
                                        let lr = (py_off + r) / 4;
                                        let lc = (px_off + c) / 4;
                                        if let Some(blk) = BLOCK_INDEX_TO_OFFSET
                                            .iter()
                                            .position(|&(br, bc)| br / 4 == lr && bc / 4 == lc)
                                        {
                                            ref_idx_store_l0[mb_idx * 16 + blk] = part_ref_l0[p];
                                        }
                                    }
                                }
                            }
                            for p in 0..2 {
                                if pred_flags[p].1 {
                                    if header.num_ref_idx_l1_active > 1 {
                                        let (py_off, px_off) =
                                            if part_h == 8 { (p * 8, 0) } else { (0, p * 8) };
                                        let (left_ref, top_ref) = cabac_neighbor_ref(
                                            &ref_idx_store_l1,
                                            mb_idx,
                                            mb_width as usize,
                                            py_off,
                                            px_off,
                                            &mb_slice_id,
                                            this_slice_id,
                                        );
                                        part_ref_l1[p] = cr.decode_ref_idx(st, left_ref, top_ref);
                                    } else {
                                        part_ref_l1[p] = 0;
                                    }
                                }
                                // Write L1 ref_idx immediately for neighbor context
                                let (py_off, px_off) =
                                    if part_h == 8 { (p * 8, 0) } else { (0, p * 8) };
                                for r in (0..part_h).step_by(4) {
                                    for c in (0..part_w).step_by(4) {
                                        let lr = (py_off + r) / 4;
                                        let lc = (px_off + c) / 4;
                                        if let Some(blk) = BLOCK_INDEX_TO_OFFSET
                                            .iter()
                                            .position(|&(br, bc)| br / 4 == lr && bc / 4 == lc)
                                        {
                                            ref_idx_store_l1[mb_idx * 16 + blk] = part_ref_l1[p];
                                        }
                                    }
                                }
                            }

                            // Parse MVDs: L0 for all partitions, then L1 for all
                            let mut mv_l0_parts = [[0i16; 2]; 2];
                            let mut mv_l1_parts = [[0i16; 2]; 2];

                            for p in 0..2 {
                                if pred_flags[p].0 {
                                    let (py_off, px_off) =
                                        if part_h == 8 { (p * 8, 0) } else { (0, p * 8) };
                                    // Store ref_idx before MV prediction
                                    for r in (0..part_h).step_by(4) {
                                        for c in (0..part_w).step_by(4) {
                                            let lr = (py_off + r) / 4;
                                            let lc = (px_off + c) / 4;
                                            if let Some(blk) = BLOCK_INDEX_TO_OFFSET
                                                .iter()
                                                .position(|&(br, bc)| br / 4 == lr && bc / 4 == lc)
                                            {
                                                ref_idx_store_l0[mb_idx * 16 + blk] =
                                                    part_ref_l0[p];
                                            }
                                        }
                                    }
                                    let amvd_x = cabac_amvd(
                                        &mvd_store,
                                        mb_idx,
                                        mb_width as usize,
                                        py_off,
                                        px_off,
                                        0,
                                        &mb_slice_id,
                                        this_slice_id,
                                    );
                                    let amvd_y = cabac_amvd(
                                        &mvd_store,
                                        mb_idx,
                                        mb_width as usize,
                                        py_off,
                                        px_off,
                                        1,
                                        &mb_slice_id,
                                        this_slice_id,
                                    );
                                    let mvd_x = cr.decode_mvd_comp(st, 40, amvd_x) as i16;
                                    let mvd_y = cr.decode_mvd_comp(st, 47, amvd_y) as i16;
                                    let (mvp_x, mvp_y) = predict_mv(
                                        &mv_store_l0,
                                        &ref_idx_store_l0,
                                        mb_idx,
                                        mb_width as usize,
                                        p,
                                        part_w,
                                        part_h,
                                        part_ref_l0[p],
                                        &mb_slice_id,
                                        this_slice_id,
                                    );
                                    mv_l0_parts[p] = [mvp_x + mvd_x, mvp_y + mvd_y];
                                    // Store MV and MVD immediately for partition 1 prediction
                                    for r in (0..part_h).step_by(4) {
                                        for c in (0..part_w).step_by(4) {
                                            let lr = (py_off + r) / 4;
                                            let lc = (px_off + c) / 4;
                                            if let Some(blk) = BLOCK_INDEX_TO_OFFSET
                                                .iter()
                                                .position(|&(br, bc)| br / 4 == lr && bc / 4 == lc)
                                            {
                                                mv_store_l0[mb_idx * 16 + blk] = mv_l0_parts[p];
                                                mvd_store[mb_idx * 16 + blk] = [mvd_x, mvd_y];
                                            }
                                        }
                                    }
                                } else {
                                    // Inactive L0: set ref_idx = -1
                                    let (py_off, px_off) =
                                        if part_h == 8 { (p * 8, 0) } else { (0, p * 8) };
                                    for r in (0..part_h).step_by(4) {
                                        for c in (0..part_w).step_by(4) {
                                            let lr = (py_off + r) / 4;
                                            let lc = (px_off + c) / 4;
                                            if let Some(blk) = BLOCK_INDEX_TO_OFFSET
                                                .iter()
                                                .position(|&(br, bc)| br / 4 == lr && bc / 4 == lc)
                                            {
                                                ref_idx_store_l0[mb_idx * 16 + blk] = -1;
                                            }
                                        }
                                    }
                                }
                            }

                            for p in 0..2 {
                                if pred_flags[p].1 {
                                    let (py_off, px_off) =
                                        if part_h == 8 { (p * 8, 0) } else { (0, p * 8) };
                                    // Store ref_idx before MV prediction
                                    for r in (0..part_h).step_by(4) {
                                        for c in (0..part_w).step_by(4) {
                                            let lr = (py_off + r) / 4;
                                            let lc = (px_off + c) / 4;
                                            if let Some(blk) = BLOCK_INDEX_TO_OFFSET
                                                .iter()
                                                .position(|&(br, bc)| br / 4 == lr && bc / 4 == lc)
                                            {
                                                ref_idx_store_l1[mb_idx * 16 + blk] =
                                                    part_ref_l1[p];
                                            }
                                        }
                                    }
                                    let amvd_x = cabac_amvd(
                                        &mvd_store_l1,
                                        mb_idx,
                                        mb_width as usize,
                                        py_off,
                                        px_off,
                                        0,
                                        &mb_slice_id,
                                        this_slice_id,
                                    );
                                    let amvd_y = cabac_amvd(
                                        &mvd_store_l1,
                                        mb_idx,
                                        mb_width as usize,
                                        py_off,
                                        px_off,
                                        1,
                                        &mb_slice_id,
                                        this_slice_id,
                                    );
                                    let mvd_x = cr.decode_mvd_comp(st, 40, amvd_x) as i16;
                                    let mvd_y = cr.decode_mvd_comp(st, 47, amvd_y) as i16;
                                    let (mvp_x, mvp_y) = predict_mv(
                                        &mv_store_l1,
                                        &ref_idx_store_l1,
                                        mb_idx,
                                        mb_width as usize,
                                        p,
                                        part_w,
                                        part_h,
                                        part_ref_l1[p],
                                        &mb_slice_id,
                                        this_slice_id,
                                    );
                                    mv_l1_parts[p] = [mvp_x + mvd_x, mvp_y + mvd_y];
                                    for r in (0..part_h).step_by(4) {
                                        for c in (0..part_w).step_by(4) {
                                            let lr = (py_off + r) / 4;
                                            let lc = (px_off + c) / 4;
                                            if let Some(blk) = BLOCK_INDEX_TO_OFFSET
                                                .iter()
                                                .position(|&(br, bc)| br / 4 == lr && bc / 4 == lc)
                                            {
                                                mv_store_l1[mb_idx * 16 + blk] = mv_l1_parts[p];
                                                mvd_store_l1[mb_idx * 16 + blk] = [mvd_x, mvd_y];
                                            }
                                        }
                                    }
                                } else {
                                    let (py_off, px_off) =
                                        if part_h == 8 { (p * 8, 0) } else { (0, p * 8) };
                                    for r in (0..part_h).step_by(4) {
                                        for c in (0..part_w).step_by(4) {
                                            let lr = (py_off + r) / 4;
                                            let lc = (px_off + c) / 4;
                                            if let Some(blk) = BLOCK_INDEX_TO_OFFSET
                                                .iter()
                                                .position(|&(br, bc)| br / 4 == lr && bc / 4 == lc)
                                            {
                                                ref_idx_store_l1[mb_idx * 16 + blk] = -1;
                                            }
                                        }
                                    }
                                }
                            }

                            // Build sub_parts for MC
                            for p in 0..2 {
                                let (py_off, px_off) =
                                    if part_h == 8 { (p * 8, 0) } else { (0, p * 8) };
                                b_sub_parts.push(BSubPart {
                                    x: px_off,
                                    y: py_off,
                                    w: part_w,
                                    h: part_h,
                                    ref_idx_l0: part_ref_l0[p],
                                    ref_idx_l1: part_ref_l1[p],
                                    mv_l0: mv_l0_parts[p],
                                    mv_l1: mv_l1_parts[p],
                                    pred_l0: pred_flags[p].0,
                                    pred_l1: pred_flags[p].1,
                                });
                            }
                        } else if raw_mb_type == 22 {
                            // B_8x8
                            let sub_mb_origins = [(0usize, 0usize), (0, 8), (8, 0), (8, 8)];
                            let mut sub_mb_types = [0u32; 4];
                            for smt in &mut sub_mb_types {
                                *smt = cr.decode_b_sub_mb_type(st);
                            }
                            // Check if any sub-partition is smaller than 8x8
                            // (sub_mb_types 0-3 are 8x8; 4+ are sub-8x8)
                            // B_Direct_8x8 (smt==0) uses 4x4 when direct_8x8_inference_flag=0
                            if sub_mb_types
                                .iter()
                                .any(|&smt| smt > 3 || (smt == 0 && !sps.direct_8x8_inference_flag))
                            {
                                no_sub_less_than_8x8_b = false;
                            }

                            // Parse ref_idx: L0 for all sub-MBs, then L1
                            let mut sub_ref_l0 = [-1i8; 4];
                            let mut sub_ref_l1 = [-1i8; 4];
                            for smb in 0..4 {
                                if sub_mb_types[smb] == 0 {
                                    // B_Direct_8x8: don't decode ref, keep -1
                                } else {
                                    let (_, _, pl0, _) = B_SUB_TABLE[sub_mb_types[smb] as usize];
                                    if pl0 {
                                        if header.num_ref_idx_l0_active > 1 {
                                            let (sy, sx) = sub_mb_origins[smb];
                                            let (left_ref, top_ref) = cabac_neighbor_ref(
                                                &ref_idx_store_l0,
                                                mb_idx,
                                                mb_width as usize,
                                                sy,
                                                sx,
                                                &mb_slice_id,
                                                this_slice_id,
                                            );
                                            sub_ref_l0[smb] =
                                                cr.decode_ref_idx(st, left_ref, top_ref);
                                        } else {
                                            sub_ref_l0[smb] = 0;
                                        }
                                    }
                                }
                                // Write L0 ref_idx immediately for neighbor context
                                let (sy, sx) = sub_mb_origins[smb];
                                for r in (0..8).step_by(4) {
                                    for c in (0..8).step_by(4) {
                                        let lr = (sy + r) / 4;
                                        let lc = (sx + c) / 4;
                                        if let Some(blk) = BLOCK_INDEX_TO_OFFSET
                                            .iter()
                                            .position(|&(br, bc)| br / 4 == lr && bc / 4 == lc)
                                        {
                                            ref_idx_store_l0[mb_idx * 16 + blk] = sub_ref_l0[smb];
                                        }
                                    }
                                }
                            }
                            for smb in 0..4 {
                                if sub_mb_types[smb] == 0 {
                                    // B_Direct_8x8: don't decode ref
                                } else {
                                    let (_, _, _, pl1) = B_SUB_TABLE[sub_mb_types[smb] as usize];
                                    if pl1 {
                                        if header.num_ref_idx_l1_active > 1 {
                                            let (sy, sx) = sub_mb_origins[smb];
                                            let (left_ref, top_ref) = cabac_neighbor_ref(
                                                &ref_idx_store_l1,
                                                mb_idx,
                                                mb_width as usize,
                                                sy,
                                                sx,
                                                &mb_slice_id,
                                                this_slice_id,
                                            );
                                            sub_ref_l1[smb] =
                                                cr.decode_ref_idx(st, left_ref, top_ref);
                                        } else {
                                            sub_ref_l1[smb] = 0;
                                        }
                                    }
                                }
                                // Write L1 ref_idx immediately for neighbor context
                                let (sy, sx) = sub_mb_origins[smb];
                                for r in (0..8).step_by(4) {
                                    for c in (0..8).step_by(4) {
                                        let lr = (sy + r) / 4;
                                        let lc = (sx + c) / 4;
                                        if let Some(blk) = BLOCK_INDEX_TO_OFFSET
                                            .iter()
                                            .position(|&(br, bc)| br / 4 == lr && bc / 4 == lc)
                                        {
                                            ref_idx_store_l1[mb_idx * 16 + blk] = sub_ref_l1[smb];
                                        }
                                    }
                                }
                            }

                            // Store ref_idx into cache for MV prediction
                            for smb in 0..4 {
                                let (sy, sx) = sub_mb_origins[smb];
                                for r in (0..8).step_by(4) {
                                    for c in (0..8).step_by(4) {
                                        let lr = (sy + r) / 4;
                                        let lc = (sx + c) / 4;
                                        if let Some(blk) = BLOCK_INDEX_TO_OFFSET
                                            .iter()
                                            .position(|&(br, bc)| br / 4 == lr && bc / 4 == lc)
                                        {
                                            ref_idx_store_l0[mb_idx * 16 + blk] = sub_ref_l0[smb];
                                            ref_idx_store_l1[mb_idx * 16 + blk] = sub_ref_l1[smb];
                                        }
                                    }
                                }
                            }

                            // Derive B_Direct_8x8 MVs per 4x4 block BEFORE MVD parsing
                            for (smb, &smt) in sub_mb_types.iter().enumerate() {
                                if smt != 0 {
                                    continue;
                                }
                                make_ctx!().derive_direct_mvs(
                                    mb_idx,
                                    smb * 4,
                                    4,
                                    header.direct_spatial_mv_pred_flag,
                                    sps.direct_8x8_inference_flag,
                                    current_poc,
                                    &_ref_pic_list_l0,
                                    &_ref_pic_list_l1,
                                );
                            }

                            // Collect sub-partition layouts
                            struct BSubLayout {
                                smb: usize,
                                sub_w: usize,
                                sub_h: usize,
                                pl0: bool,
                                pl1: bool,
                                offsets: Vec<(usize, usize)>,
                            }
                            let mut layouts: Vec<BSubLayout> = Vec::new();
                            for (smb, &smt_val) in sub_mb_types.iter().enumerate() {
                                let smt = smt_val as usize;
                                if smt == 0 {
                                    layouts.push(BSubLayout {
                                        smb,
                                        sub_w: 8,
                                        sub_h: 8,
                                        pl0: false,
                                        pl1: false,
                                        offsets: vec![(0, 0)],
                                    });
                                    continue;
                                }
                                let (sub_w, sub_h, pl0, pl1) = B_SUB_TABLE[smt];
                                let offsets = match (sub_w, sub_h) {
                                    (8, 8) => vec![(0, 0)],
                                    (8, 4) => vec![(0, 0), (0, 4)],
                                    (4, 8) => vec![(0, 0), (4, 0)],
                                    (4, 4) => vec![(0, 0), (4, 0), (0, 4), (4, 4)],
                                    _ => unreachable!(),
                                };
                                layouts.push(BSubLayout {
                                    smb,
                                    sub_w,
                                    sub_h,
                                    pl0,
                                    pl1,
                                    offsets,
                                });
                            }

                            // Parse L0 MVDs
                            for layout in &layouts {
                                if sub_mb_types[layout.smb] == 0 {
                                    continue;
                                }
                                if !layout.pl0 {
                                    continue;
                                }
                                let (sy, sx) = sub_mb_origins[layout.smb];
                                for &(dx, dy) in &layout.offsets {
                                    let px = sx + dx;
                                    let py = sy + dy;
                                    let amvd_x = cabac_amvd(
                                        &mvd_store,
                                        mb_idx,
                                        mb_width as usize,
                                        py,
                                        px,
                                        0,
                                        &mb_slice_id,
                                        this_slice_id,
                                    );
                                    let amvd_y = cabac_amvd(
                                        &mvd_store,
                                        mb_idx,
                                        mb_width as usize,
                                        py,
                                        px,
                                        1,
                                        &mb_slice_id,
                                        this_slice_id,
                                    );
                                    let mvd_x = cr.decode_mvd_comp(st, 40, amvd_x) as i16;
                                    let mvd_y = cr.decode_mvd_comp(st, 47, amvd_y) as i16;
                                    let (mvp_x, mvp_y) = predict_mv_sub(
                                        &mv_store_l0,
                                        &ref_idx_store_l0,
                                        mb_idx,
                                        mb_width as usize,
                                        px,
                                        py,
                                        layout.sub_w,
                                        layout.sub_h,
                                        sub_ref_l0[layout.smb],
                                        &mb_slice_id,
                                        this_slice_id,
                                    );
                                    let mv = [mvp_x + mvd_x, mvp_y + mvd_y];
                                    for r in (0..layout.sub_h).step_by(4) {
                                        for c in (0..layout.sub_w).step_by(4) {
                                            let lr = (py + r) / 4;
                                            let lc = (px + c) / 4;
                                            if let Some(blk) = BLOCK_INDEX_TO_OFFSET
                                                .iter()
                                                .position(|&(br, bc)| br / 4 == lr && bc / 4 == lc)
                                            {
                                                mv_store_l0[mb_idx * 16 + blk] = mv;
                                                mvd_store[mb_idx * 16 + blk] = [mvd_x, mvd_y];
                                            }
                                        }
                                    }
                                }
                            }

                            // Parse L1 MVDs
                            for layout in &layouts {
                                if sub_mb_types[layout.smb] == 0 {
                                    continue;
                                }
                                if !layout.pl1 {
                                    continue;
                                }
                                let (sy, sx) = sub_mb_origins[layout.smb];
                                for &(dx, dy) in &layout.offsets {
                                    let px = sx + dx;
                                    let py = sy + dy;
                                    let amvd_x = cabac_amvd(
                                        &mvd_store_l1,
                                        mb_idx,
                                        mb_width as usize,
                                        py,
                                        px,
                                        0,
                                        &mb_slice_id,
                                        this_slice_id,
                                    );
                                    let amvd_y = cabac_amvd(
                                        &mvd_store_l1,
                                        mb_idx,
                                        mb_width as usize,
                                        py,
                                        px,
                                        1,
                                        &mb_slice_id,
                                        this_slice_id,
                                    );
                                    let mvd_x = cr.decode_mvd_comp(st, 40, amvd_x) as i16;
                                    let mvd_y = cr.decode_mvd_comp(st, 47, amvd_y) as i16;
                                    let (mvp_x, mvp_y) = predict_mv_sub(
                                        &mv_store_l1,
                                        &ref_idx_store_l1,
                                        mb_idx,
                                        mb_width as usize,
                                        px,
                                        py,
                                        layout.sub_w,
                                        layout.sub_h,
                                        sub_ref_l1[layout.smb],
                                        &mb_slice_id,
                                        this_slice_id,
                                    );
                                    let mv = [mvp_x + mvd_x, mvp_y + mvd_y];
                                    for r in (0..layout.sub_h).step_by(4) {
                                        for c in (0..layout.sub_w).step_by(4) {
                                            let lr = (py + r) / 4;
                                            let lc = (px + c) / 4;
                                            if let Some(blk) = BLOCK_INDEX_TO_OFFSET
                                                .iter()
                                                .position(|&(br, bc)| br / 4 == lr && bc / 4 == lc)
                                            {
                                                mv_store_l1[mb_idx * 16 + blk] = mv;
                                                mvd_store_l1[mb_idx * 16 + blk] = [mvd_x, mvd_y];
                                            }
                                        }
                                    }
                                }
                            }

                            // Build sub_parts from stored MVs
                            for layout in &layouts {
                                let (sy, sx) = sub_mb_origins[layout.smb];
                                if sub_mb_types[layout.smb] == 0 {
                                    // B_Direct_8x8: per-4x4-block sub_parts
                                    let base = mb_idx * 16;
                                    for r in (0..8).step_by(4) {
                                        for c in (0..8).step_by(4) {
                                            let lr = (sy + r) / 4;
                                            let lc = (sx + c) / 4;
                                            let blk = BLOCK_INDEX_TO_OFFSET
                                                .iter()
                                                .position(|&(br, bc)| br / 4 == lr && bc / 4 == lc)
                                                .unwrap_or(0);
                                            let ri_l0 = ref_idx_store_l0[base + blk];
                                            let ri_l1 = ref_idx_store_l1[base + blk];
                                            b_sub_parts.push(BSubPart {
                                                x: sx + c,
                                                y: sy + r,
                                                w: 4,
                                                h: 4,
                                                ref_idx_l0: ri_l0,
                                                ref_idx_l1: ri_l1,
                                                mv_l0: mv_store_l0[base + blk],
                                                mv_l1: mv_store_l1[base + blk],
                                                pred_l0: ri_l0 >= 0,
                                                pred_l1: ri_l1 >= 0,
                                            });
                                        }
                                    }
                                } else {
                                    let smt = sub_mb_types[layout.smb] as usize;
                                    let (_, _, pl0, pl1) = B_SUB_TABLE[smt];
                                    for &(dx, dy) in &layout.offsets {
                                        let px = sx + dx;
                                        let py = sy + dy;
                                        let blk0 = BLOCK_INDEX_TO_OFFSET
                                            .iter()
                                            .position(|&(br, bc)| {
                                                br / 4 == py / 4 && bc / 4 == px / 4
                                            })
                                            .unwrap_or(0);
                                        let base = mb_idx * 16;
                                        b_sub_parts.push(BSubPart {
                                            x: px,
                                            y: py,
                                            w: layout.sub_w,
                                            h: layout.sub_h,
                                            ref_idx_l0: sub_ref_l0[layout.smb],
                                            ref_idx_l1: sub_ref_l1[layout.smb],
                                            mv_l0: mv_store_l0[base + blk0],
                                            mv_l1: mv_store_l1[base + blk0],
                                            pred_l0: pl0,
                                            pred_l1: pl1,
                                        });
                                    }
                                }
                            }
                        } else {
                            return Err(DecodeError::InvalidSyntax(
                                "invalid CABAC B-slice mb_type",
                            ));
                        }

                        // Motion compensation for all sub-parts
                        for sp in &b_sub_parts {
                            let abs_x = mb_x + sp.x;
                            let abs_y = mb_y + sp.y;
                            let mut luma_pred = vec![0u8; sp.w * sp.h];

                            if sp.pred_l0 && sp.pred_l1 {
                                let mut p0 = vec![0u8; sp.w * sp.h];
                                let mut p1 = vec![0u8; sp.w * sp.h];
                                inter_pred::luma_mc(
                                    ref_pic_safe(&_ref_pic_list_l0, sp.ref_idx_l0),
                                    abs_x as i32,
                                    abs_y as i32,
                                    sp.mv_l0[0] as i32,
                                    sp.mv_l0[1] as i32,
                                    sp.w,
                                    sp.h,
                                    &mut p0,
                                );
                                inter_pred::luma_mc(
                                    ref_pic_safe(&_ref_pic_list_l1, sp.ref_idx_l1),
                                    abs_x as i32,
                                    abs_y as i32,
                                    sp.mv_l1[0] as i32,
                                    sp.mv_l1[1] as i32,
                                    sp.w,
                                    sp.h,
                                    &mut p1,
                                );
                                wctx.apply_bi(
                                    &p0,
                                    &p1,
                                    &mut luma_pred,
                                    sp.ref_idx_l0 as usize,
                                    sp.ref_idx_l1 as usize,
                                    false,
                                    0,
                                );
                            } else if sp.pred_l0 {
                                inter_pred::luma_mc(
                                    ref_pic_safe(&_ref_pic_list_l0, sp.ref_idx_l0),
                                    abs_x as i32,
                                    abs_y as i32,
                                    sp.mv_l0[0] as i32,
                                    sp.mv_l0[1] as i32,
                                    sp.w,
                                    sp.h,
                                    &mut luma_pred,
                                );
                                if use_weight == 1 {
                                    wctx.apply_uni(
                                        &mut luma_pred,
                                        0,
                                        sp.ref_idx_l0 as usize,
                                        false,
                                        0,
                                    );
                                }
                            } else if sp.pred_l1 {
                                inter_pred::luma_mc(
                                    ref_pic_safe(&_ref_pic_list_l1, sp.ref_idx_l1),
                                    abs_x as i32,
                                    abs_y as i32,
                                    sp.mv_l1[0] as i32,
                                    sp.mv_l1[1] as i32,
                                    sp.w,
                                    sp.h,
                                    &mut luma_pred,
                                );
                                if use_weight == 1 {
                                    wctx.apply_uni(
                                        &mut luma_pred,
                                        1,
                                        sp.ref_idx_l1 as usize,
                                        false,
                                        0,
                                    );
                                }
                            }

                            for r in 0..sp.h {
                                for c in 0..sp.w {
                                    frame.y[(abs_y + r) * stride + abs_x + c] =
                                        luma_pred[r * sp.w + c];
                                }
                            }
                            // Chroma MC
                            let cw = (width / 2) as usize;
                            let chroma_h = (height / 2) as usize;
                            let cx = abs_x / 2;
                            let cy = abs_y / 2;
                            let chw = sp.w.max(2) / 2;
                            let chh = sp.h.max(2) / 2;

                            for plane_idx in 0..2 {
                                let mut chroma_pred = vec![0u8; chw * chh];
                                if sp.pred_l0 && sp.pred_l1 {
                                    let ref_l0 = ref_pic_safe(&_ref_pic_list_l0, sp.ref_idx_l0);
                                    let ref_l1 = ref_pic_safe(&_ref_pic_list_l1, sp.ref_idx_l1);
                                    let cr0 = if plane_idx == 0 { &ref_l0.u } else { &ref_l0.v };
                                    let cr1 = if plane_idx == 0 { &ref_l1.u } else { &ref_l1.v };
                                    let mut c0 = vec![0u8; chw * chh];
                                    let mut c1 = vec![0u8; chw * chh];
                                    inter_pred::chroma_mc(
                                        cr0,
                                        cw,
                                        chroma_h,
                                        cx as i32,
                                        cy as i32,
                                        sp.mv_l0[0] as i32,
                                        sp.mv_l0[1] as i32,
                                        chw,
                                        chh,
                                        &mut c0,
                                    );
                                    inter_pred::chroma_mc(
                                        cr1,
                                        cw,
                                        chroma_h,
                                        cx as i32,
                                        cy as i32,
                                        sp.mv_l1[0] as i32,
                                        sp.mv_l1[1] as i32,
                                        chw,
                                        chh,
                                        &mut c1,
                                    );
                                    wctx.apply_bi(
                                        &c0,
                                        &c1,
                                        &mut chroma_pred,
                                        sp.ref_idx_l0 as usize,
                                        sp.ref_idx_l1 as usize,
                                        true,
                                        plane_idx,
                                    );
                                } else if sp.pred_l0 {
                                    let ref_pic = ref_pic_safe(&_ref_pic_list_l0, sp.ref_idx_l0);
                                    let plane = if plane_idx == 0 {
                                        &ref_pic.u
                                    } else {
                                        &ref_pic.v
                                    };
                                    inter_pred::chroma_mc(
                                        plane,
                                        cw,
                                        chroma_h,
                                        cx as i32,
                                        cy as i32,
                                        sp.mv_l0[0] as i32,
                                        sp.mv_l0[1] as i32,
                                        chw,
                                        chh,
                                        &mut chroma_pred,
                                    );
                                    if use_weight == 1 {
                                        wctx.apply_uni(
                                            &mut chroma_pred,
                                            0,
                                            sp.ref_idx_l0 as usize,
                                            true,
                                            plane_idx,
                                        );
                                    }
                                } else if sp.pred_l1 {
                                    let ref_pic = ref_pic_safe(&_ref_pic_list_l1, sp.ref_idx_l1);
                                    let plane = if plane_idx == 0 {
                                        &ref_pic.u
                                    } else {
                                        &ref_pic.v
                                    };
                                    inter_pred::chroma_mc(
                                        plane,
                                        cw,
                                        chroma_h,
                                        cx as i32,
                                        cy as i32,
                                        sp.mv_l1[0] as i32,
                                        sp.mv_l1[1] as i32,
                                        chw,
                                        chh,
                                        &mut chroma_pred,
                                    );
                                }

                                let fp = if plane_idx == 0 {
                                    &mut frame.u
                                } else {
                                    &mut frame.v
                                };
                                for r in 0..chh {
                                    for c in 0..chw {
                                        fp[(cy + r) * cw + cx + c] = chroma_pred[r * chw + c];
                                    }
                                }
                            }
                        }

                        // Residual: CABAC CBP + coefficients
                        let left_cbp_raw = if !mb_idx.is_multiple_of(mb_width as usize)
                            && mb_slice_id[mb_idx - 1] == this_slice_id
                        {
                            mb_cbp[mb_idx - 1]
                        } else {
                            0x00Fu16
                        };
                        let top_cbp_raw = if mb_idx >= mb_width as usize
                            && mb_slice_id[mb_idx - mb_width as usize] == this_slice_id
                        {
                            mb_cbp[mb_idx - mb_width as usize]
                        } else {
                            0x00Fu16
                        };
                        let left_cbp = ((left_cbp_raw & 0x7F0)
                            | (left_cbp_raw & 2)
                            | (((left_cbp_raw >> 2) & 2) << 2))
                            as u8;
                        let top_cbp = top_cbp_raw as u8;
                        let cbp_luma = cr.decode_cbp_luma(st, left_cbp, top_cbp);
                        let left_cbp_c = ((left_cbp_raw >> 4) & 3) as u8;
                        let top_cbp_c = ((top_cbp_raw >> 4) & 3) as u8;
                        let cbp_chroma = cr.decode_cbp_chroma(st, left_cbp_c, top_cbp_c);
                        mb_cbp[mb_idx] = (cbp_luma as u16) | ((cbp_chroma as u16) << 4);

                        // 8x8 transform flag for B inter MBs (spec 7.3.5).
                        // Only present when noSubMbPartSizeLessThan8x8Flag is true.
                        let use_8x8_b_inter = if pps.transform_8x8_mode_flag
                            && cbp_luma != 0
                            && no_sub_less_than_8x8_b
                        {
                            let nts = {
                                let left = if !mb_idx.is_multiple_of(mb_width as usize)
                                    && mb_slice_id[mb_idx - 1] == this_slice_id
                                {
                                    mb_is_8x8dct[mb_idx - 1] as usize
                                } else {
                                    0
                                };
                                let top = if mb_idx >= mb_width as usize
                                    && mb_slice_id[mb_idx - mb_width as usize] == this_slice_id
                                {
                                    mb_is_8x8dct[mb_idx - mb_width as usize] as usize
                                } else {
                                    0
                                };
                                left + top
                            };
                            cr.get_cabac(&mut st[399 + nts]) != 0
                        } else {
                            false
                        };
                        mb_is_8x8dct[mb_idx] = use_8x8_b_inter;

                        let qp_y = if cbp_luma != 0 || cbp_chroma != 0 {
                            let delta = cr.decode_mb_qp_delta(st, last_qp_delta_nonzero);
                            last_qp_delta_nonzero = delta != 0;
                            ((prev_mb_qp + delta + 52) % 52 + 52) % 52
                        } else {
                            last_qp_delta_nonzero = false;
                            prev_mb_qp
                        };
                        prev_mb_qp = qp_y;

                        // Luma residual
                        if cbp_luma != 0 {
                            let mut luma_residual = [0i32; 256];
                            if use_8x8_b_inter {
                                let scale_8x8 = &pps.scaling_list_8x8[1];
                                for i8x8 in 0..4 {
                                    if cbp_luma & (1 << i8x8) == 0 {
                                        continue;
                                    }
                                    // cat=5: no coded_block_flag, CBP bit is sufficient
                                    let (coeffs, tc) = cr.decode_residual_cabac(st, 5, 64);
                                    let tc_per = tc.div_ceil(4);
                                    for sub in 0..4 {
                                        nc_luma[mb_idx * 16 + i8x8 * 4 + sub] = tc_per;
                                    }
                                    let mut block_8x8 = [0i32; 64];
                                    for (pos, val) in &coeffs {
                                        block_8x8[ZIGZAG_8X8_CABAC[*pos]] = *val;
                                    }
                                    dequant_8x8(&mut block_8x8, qp_y, scale_8x8);
                                    inverse_dct_8x8(&mut block_8x8);
                                    let row_off = (i8x8 / 2) * 8;
                                    let col_off = (i8x8 % 2) * 8;
                                    for r in 0..8 {
                                        for c in 0..8 {
                                            luma_residual[(row_off + r) * 16 + col_off + c] =
                                                block_8x8[r * 8 + c];
                                        }
                                    }
                                }
                            } else {
                                for blk in 0..16 {
                                    if cbp_luma & (1 << (blk / 4)) != 0 {
                                        let left_nz = cabac_neighbor_nz_luma(
                                            &nc_luma,
                                            mb_idx,
                                            mb_width as usize,
                                            blk,
                                            true,
                                            false,
                                            &mb_slice_id,
                                            this_slice_id,
                                        );
                                        let top_nz = cabac_neighbor_nz_luma(
                                            &nc_luma,
                                            mb_idx,
                                            mb_width as usize,
                                            blk,
                                            false,
                                            false,
                                            &mb_slice_id,
                                            this_slice_id,
                                        );
                                        if cr.decode_coded_block_flag(st, 2, left_nz, top_nz) {
                                            let (coeffs, tc) = cr.decode_residual_cabac(st, 2, 16);
                                            nc_luma[mb_idx * 16 + blk] = tc;
                                            let mut block_coeffs = [0i32; 16];
                                            for (pos, val) in &coeffs {
                                                let (r, c) = ZIGZAG_4X4[*pos];
                                                block_coeffs[r * 4 + c] = *val;
                                            }
                                            dequant_4x4_full(
                                                &mut block_coeffs,
                                                qp_y,
                                                &pps.scaling_list_4x4[3],
                                            );
                                            inverse_dct_4x4(&mut block_coeffs);
                                            let (blk_row, blk_col) = BLOCK_INDEX_TO_OFFSET[blk];
                                            for r in 0..4 {
                                                for c in 0..4 {
                                                    luma_residual
                                                        [(blk_row + r) * 16 + blk_col + c] =
                                                        block_coeffs[r * 4 + c];
                                                }
                                            }
                                        }
                                    }
                                }
                            } // close if use_8x8_b_inter else
                            for r in 0..16 {
                                for c in 0..16 {
                                    let val = (frame.y[(mb_y + r) * stride + mb_x + c] as i32
                                        + luma_residual[r * 16 + c])
                                        .clamp(0, 255)
                                        as u8;
                                    frame.y[(mb_y + r) * stride + mb_x + c] = val;
                                }
                            }
                        }

                        // Chroma residual
                        let qp_c = chroma_qp(qp_y, pps.chroma_qp_index_offset);
                        if cbp_chroma >= 1 {
                            let mut chroma_dc_cb = [0i32; 4];
                            let mut chroma_dc_cr = [0i32; 4];
                            let left_dc_nz = if !mb_idx.is_multiple_of(mb_width as usize)
                                && mb_slice_id[mb_idx - 1] == this_slice_id
                            {
                                (mb_cbp[mb_idx - 1] >> 6) & 1 != 0
                            } else {
                                false
                            };
                            let top_dc_nz = if mb_idx >= mb_width as usize
                                && mb_slice_id[mb_idx - mb_width as usize] == this_slice_id
                            {
                                (mb_cbp[mb_idx - mb_width as usize] >> 6) & 1 != 0
                            } else {
                                false
                            };
                            if cr.decode_coded_block_flag(st, 3, left_dc_nz, top_dc_nz) {
                                let (coeffs, _) = cr.decode_residual_cabac(st, 3, 4);
                                for (pos, val) in coeffs {
                                    chroma_dc_cb[pos] = val;
                                }
                                mb_cbp[mb_idx] |= 0x40;
                            }
                            let left_dc_cr = if !mb_idx.is_multiple_of(mb_width as usize)
                                && mb_slice_id[mb_idx - 1] == this_slice_id
                            {
                                (mb_cbp[mb_idx - 1] >> 7) & 1 != 0
                            } else {
                                false
                            };
                            let top_dc_cr = if mb_idx >= mb_width as usize
                                && mb_slice_id[mb_idx - mb_width as usize] == this_slice_id
                            {
                                (mb_cbp[mb_idx - mb_width as usize] >> 7) & 1 != 0
                            } else {
                                false
                            };
                            if cr.decode_coded_block_flag(st, 3, left_dc_cr, top_dc_cr) {
                                let (coeffs, _) = cr.decode_residual_cabac(st, 3, 4);
                                for (pos, val) in coeffs {
                                    chroma_dc_cr[pos] = val;
                                }
                                mb_cbp[mb_idx] |= 0x80;
                            }

                            let chroma_width = (width / 2) as usize;
                            let chroma_mb_x = mb_x / 2;
                            let chroma_mb_y = mb_y / 2;
                            for (plane_dc, frame_plane, scale_idx) in [
                                (&mut chroma_dc_cb, &mut frame.u, 4usize),
                                (&mut chroma_dc_cr, &mut frame.v, 5usize),
                            ] {
                                let chroma_scale = &pps.scaling_list_4x4[scale_idx];
                                inverse_hadamard_2x2(plane_dc);
                                dequant_chroma_dc(plane_dc, qp_c, chroma_scale[0]);
                                let mut chroma_residual = [0i32; 64];
                                for blk in 0..4 {
                                    let blk_row = (blk / 2) * 4;
                                    let blk_col = (blk % 2) * 4;
                                    let mut block_raster = [0i32; 16];
                                    block_raster[0] = plane_dc[blk];
                                    if cbp_chroma >= 2 {
                                        let nc_arr = if scale_idx == 4 { &nc_cb } else { &nc_cr };
                                        let left_nz = cabac_neighbor_nz_chroma(
                                            nc_arr,
                                            mb_idx,
                                            mb_width as usize,
                                            blk,
                                            true,
                                            false,
                                            &mb_slice_id,
                                            this_slice_id,
                                        );
                                        let top_nz = cabac_neighbor_nz_chroma(
                                            nc_arr,
                                            mb_idx,
                                            mb_width as usize,
                                            blk,
                                            false,
                                            false,
                                            &mb_slice_id,
                                            this_slice_id,
                                        );
                                        if cr.decode_coded_block_flag(st, 4, left_nz, top_nz) {
                                            let (coeffs, tc) = cr.decode_residual_cabac(st, 4, 15);
                                            if scale_idx == 4 {
                                                nc_cb[mb_idx * 4 + blk] = tc;
                                            } else {
                                                nc_cr[mb_idx * 4 + blk] = tc;
                                            }
                                            for (pos, val) in coeffs {
                                                let (r, c) = ZIGZAG_4X4[pos + 1];
                                                block_raster[r * 4 + c] = val;
                                            }
                                            dequant_4x4_ac_raster(
                                                &mut block_raster,
                                                qp_c,
                                                chroma_scale,
                                            );
                                        }
                                    }
                                    inverse_dct_4x4(&mut block_raster);
                                    for r in 0..4 {
                                        for c in 0..4 {
                                            chroma_residual[(blk_row + r) * 8 + blk_col + c] =
                                                block_raster[r * 4 + c];
                                        }
                                    }
                                }
                                for y in 0..8 {
                                    for x in 0..8 {
                                        let val = (frame_plane
                                            [(chroma_mb_y + y) * chroma_width + chroma_mb_x + x]
                                            as i32
                                            + chroma_residual[y * 8 + x])
                                            .clamp(0, 255)
                                            as u8;
                                        frame_plane
                                            [(chroma_mb_y + y) * chroma_width + chroma_mb_x + x] =
                                            val;
                                    }
                                }
                            }
                        }

                        mb_info[mb_idx] = MbInfo {
                            mb_type: MbType::Inter,
                            qp_y,
                            ..Default::default()
                        };
                        mb_idx += 1;
                        continue;
                    }
                }

                // I-slice CABAC: decode mb_type
                // Context depends on whether neighbors are I16x16 (not I4x4)
                let left_is_i16 = if !mb_idx.is_multiple_of(mb_width as usize)
                    && mb_slice_id[mb_idx - 1] == this_slice_id
                {
                    is_i16x16[mb_idx - 1]
                } else {
                    false
                };
                let top_is_i16 = if mb_idx >= mb_width as usize
                    && mb_slice_id[mb_idx - mb_width as usize] == this_slice_id
                {
                    is_i16x16[mb_idx - mb_width as usize]
                } else {
                    false
                };
                let mb_type = cr.decode_intra_mb_type(st, 3, left_is_i16, top_is_i16, true);

                // Cross-slice intra prediction: neighbors from other slices unavailable (spec 6.4.1)
                let above_mb_avail_i = mb_idx >= mb_width as usize
                    && mb_slice_id[mb_idx - mb_width as usize] == this_slice_id;
                let left_mb_avail_i = !mb_idx.is_multiple_of(mb_width as usize)
                    && mb_slice_id[mb_idx - 1] == this_slice_id;
                let above_left_mb_avail_i = mb_idx >= mb_width as usize
                    && !mb_idx.is_multiple_of(mb_width as usize)
                    && mb_slice_id[mb_idx - mb_width as usize - 1] == this_slice_id;

                // I_PCM via CABAC
                if mb_type == 25 {
                    let pcm_pos = cr.pcm_byte_position();
                    let pcm_data = &nal.rbsp[pcm_pos..];
                    let mut off = 0;
                    for r in 0..16 {
                        for c in 0..16 {
                            frame.y[(mb_y + r) * stride + mb_x + c] = pcm_data[off];
                            off += 1;
                        }
                    }
                    let cw = (width / 2) as usize;
                    let cx = mb_x / 2;
                    let cy = mb_y / 2;
                    for r in 0..8 {
                        for c in 0..8 {
                            frame.u[(cy + r) * cw + cx + c] = pcm_data[off];
                            off += 1;
                        }
                    }
                    for r in 0..8 {
                        for c in 0..8 {
                            frame.v[(cy + r) * cw + cx + c] = pcm_data[off];
                            off += 1;
                        }
                    }
                    for blk in 0..16 {
                        nc_luma[mb_idx * 16 + blk] = 16;
                    }
                    for blk in 0..4 {
                        nc_cb[mb_idx * 4 + blk] = 16;
                        nc_cr[mb_idx * 4 + blk] = 16;
                    }
                    cr.reinit(pcm_pos + off);
                    prev_mb_qp = 0;
                    last_qp_delta_nonzero = false;
                    is_i16x16[mb_idx] = false;
                    mb_cbp[mb_idx] = 0;
                    mb_info[mb_idx] = MbInfo {
                        mb_type: MbType::Ipcm,
                        qp_y: 0,
                        ..Default::default()
                    };
                    mb_idx += 1;
                    continue;
                }

                if mb_type == 0 {
                    // I4x4/I8x8 via CABAC
                    let nts = {
                        let left = if !mb_idx.is_multiple_of(mb_width as usize)
                            && mb_slice_id[mb_idx - 1] == this_slice_id
                        {
                            mb_is_8x8dct[mb_idx - 1] as usize
                        } else {
                            0
                        };
                        let top = if mb_idx >= mb_width as usize
                            && mb_slice_id[mb_idx - mb_width as usize] == this_slice_id
                        {
                            mb_is_8x8dct[mb_idx - mb_width as usize] as usize
                        } else {
                            0
                        };
                        left + top
                    };
                    let use_8x8_intra =
                        pps.transform_8x8_mode_flag && cr.get_cabac(&mut st[399 + nts]) != 0;
                    mb_is_8x8dct[mb_idx] = use_8x8_intra;

                    // Parse prediction modes: 4 for I8x8, 16 for I4x4
                    let num_modes = if use_8x8_intra { 4 } else { 16 };
                    let mut pred_modes = [2u8; 16];
                    for blk_idx in 0..num_modes {
                        let blk = if use_8x8_intra { blk_idx * 4 } else { blk_idx };
                        let predicted = predict_i4x4_mode(
                            &i4x4_modes,
                            mb_idx,
                            mb_width as usize,
                            blk,
                            &mb_slice_id,
                            this_slice_id,
                        );
                        let mode = cr.decode_intra4x4_pred_mode(st, predicted);
                        if use_8x8_intra {
                            for sub in 0..4 {
                                pred_modes[blk_idx * 4 + sub] = mode;
                                i4x4_modes[mb_idx * 16 + blk_idx * 4 + sub] = mode;
                            }
                        } else {
                            pred_modes[blk] = mode;
                            i4x4_modes[mb_idx * 16 + blk] = mode;
                        }
                    }

                    let left_cm = if !mb_idx.is_multiple_of(mb_width as usize)
                        && mb_slice_id[mb_idx - 1] == this_slice_id
                    {
                        mb_chroma_pred[mb_idx - 1]
                    } else {
                        0
                    };
                    let top_cm = if mb_idx >= mb_width as usize
                        && mb_slice_id[mb_idx - mb_width as usize] == this_slice_id
                    {
                        mb_chroma_pred[mb_idx - mb_width as usize]
                    } else {
                        0
                    };
                    let intra_chroma_pred_mode = cr.decode_chroma_pred_mode(st, left_cm, top_cm);
                    mb_chroma_pred[mb_idx] = intra_chroma_pred_mode;

                    // CBP with proper neighbor context
                    let unavail_cbp: u16 = 0x7CF;
                    let left_cbp_raw = if !mb_idx.is_multiple_of(mb_width as usize)
                        && mb_slice_id[mb_idx - 1] == this_slice_id
                    {
                        mb_cbp[mb_idx - 1]
                    } else {
                        unavail_cbp
                    };
                    let top_cbp_raw = if mb_idx >= mb_width as usize
                        && mb_slice_id[mb_idx - mb_width as usize] == this_slice_id
                    {
                        mb_cbp[mb_idx - mb_width as usize]
                    } else {
                        unavail_cbp
                    };
                    // For left: extract right-side 8x8 block bits (bits 1,3 of luma)
                    // plus upper bits (chroma + DC flags)
                    let left_cbp = ((left_cbp_raw & 0x7F0)
                        | (left_cbp_raw & 2)
                        | (((left_cbp_raw >> 2) & 2) << 2))
                        as u8;
                    let top_cbp = top_cbp_raw as u8;
                    let cbp_luma = cr.decode_cbp_luma(st, left_cbp, top_cbp);
                    let left_cbp_c = ((left_cbp_raw >> 4) & 3) as u8;
                    let top_cbp_c = ((top_cbp_raw >> 4) & 3) as u8;
                    let cbp_chroma = cr.decode_cbp_chroma(st, left_cbp_c, top_cbp_c);
                    mb_cbp[mb_idx] = (cbp_luma as u16) | ((cbp_chroma as u16) << 4);
                    let qp_y = if cbp_luma != 0 || cbp_chroma != 0 {
                        let delta = cr.decode_mb_qp_delta(st, last_qp_delta_nonzero);
                        last_qp_delta_nonzero = delta != 0;
                        ((prev_mb_qp + delta + 52) % 52 + 52) % 52
                    } else {
                        last_qp_delta_nonzero = false;
                        prev_mb_qp
                    };
                    prev_mb_qp = qp_y;
                    let _qp_c = chroma_qp(qp_y, pps.chroma_qp_index_offset);
                    let above_mb_avail = above_mb_avail_i;
                    let left_mb_avail = left_mb_avail_i;
                    let above_left_mb_avail = above_left_mb_avail_i;
                    let above_right_mb_avail = mb_idx >= mb_width as usize
                        && (mb_idx % mb_width as usize) + 1 < mb_width as usize
                        && mb_slice_id[mb_idx - mb_width as usize + 1] == this_slice_id;
                    if use_8x8_intra {
                        // I8x8 via CABAC: decode 4 blocks of 64 coefficients
                        let mut luma_residual = [0i32; 256];
                        for i8x8 in 0..4 {
                            if cbp_luma & (1 << i8x8) == 0 {
                                // Set nC=0 for all sub-blocks
                                for sub in 0..4 {
                                    nc_luma[mb_idx * 16 + i8x8 * 4 + sub] = 0;
                                }
                                continue;
                            }
                            // cat=5: no coded_block_flag, CBP bit is sufficient
                            let (coeffs, tc) = cr.decode_residual_cabac(st, 5, 64);
                            // Distribute nC across sub-blocks
                            let tc_per = tc.div_ceil(4);
                            for sub in 0..4 {
                                nc_luma[mb_idx * 16 + i8x8 * 4 + sub] = tc_per;
                            }
                            let mut block_8x8 = [0i32; 64];
                            for (pos, val) in &coeffs {
                                block_8x8[ZIGZAG_8X8_CABAC[*pos]] = *val;
                            }
                            dequant_8x8(&mut block_8x8, qp_y, &pps.scaling_list_8x8[0]);
                            inverse_dct_8x8(&mut block_8x8);
                            let row_off = (i8x8 / 2) * 8;
                            let col_off = (i8x8 % 2) * 8;
                            for r in 0..8 {
                                for c in 0..8 {
                                    luma_residual[(row_off + r) * 16 + col_off + c] =
                                        block_8x8[r * 8 + c];
                                }
                            }
                        }
                        // I8x8 prediction + residual reconstruction
                        for i8x8 in 0..4 {
                            let row_off = (i8x8 / 2) * 8;
                            let col_off = (i8x8 % 2) * 8;
                            let px = mb_x + col_off;
                            let py = mb_y + row_off;
                            let above_avail_8 = py > 0 && (row_off > 0 || above_mb_avail);
                            let above_buf: Option<[u8; 16]> = if above_avail_8 {
                                let mut buf = [0u8; 16];
                                for (i, b) in buf.iter_mut().enumerate().take(8) {
                                    *b = frame.y[(py - 1) * stride + px + i];
                                }
                                let has_tr = if row_off == 0 {
                                    if px + 8 < (mb_x + 16).min(stride) {
                                        true
                                    } else if px + 8 < stride {
                                        above_right_mb_avail
                                    } else {
                                        false
                                    }
                                } else {
                                    col_off == 0
                                };
                                if has_tr {
                                    for i in 0..8 {
                                        let col = (px + 8 + i).min(stride - 1);
                                        buf[8 + i] = frame.y[(py - 1) * stride + col];
                                    }
                                } else {
                                    let last = buf[7];
                                    buf[8..].fill(last);
                                }
                                Some(buf)
                            } else {
                                None
                            };
                            let left_avail_8 = px > 0 && (col_off > 0 || left_mb_avail);
                            let left_buf: Option<[u8; 8]> = if left_avail_8 {
                                let mut buf = [0u8; 8];
                                for (i, b) in buf.iter_mut().enumerate() {
                                    *b = frame.y[(py + i) * stride + px - 1];
                                }
                                Some(buf)
                            } else {
                                None
                            };
                            let al_avail_8 = px > 0
                                && py > 0
                                && ((row_off > 0 && col_off > 0)
                                    || (row_off > 0 && col_off == 0 && left_mb_avail)
                                    || (row_off == 0 && col_off > 0 && above_mb_avail)
                                    || (row_off == 0 && col_off == 0 && above_left_mb_avail));
                            let above_left_val = if al_avail_8 {
                                Some(frame.y[(py - 1) * stride + px - 1])
                            } else {
                                None
                            };
                            let has_topright = above_buf.is_some()
                                && ((row_off == 0 && px + 8 < stride)
                                    || (row_off > 0 && col_off == 0));
                            let mode = pred_modes[i8x8 * 4];
                            let mut pred = [0u8; 64];
                            predict_intra_8x8(
                                mode,
                                above_buf.as_ref().map(|b| &b[..]),
                                left_buf.as_ref().map(|b| &b[..]),
                                above_left_val,
                                has_topright,
                                &mut pred,
                            );
                            for r in 0..8 {
                                for c in 0..8 {
                                    let val = (pred[r * 8 + c] as i32
                                        + luma_residual[(row_off + r) * 16 + col_off + c])
                                        .clamp(0, 255)
                                        as u8;
                                    frame.y[(py + r) * stride + px + c] = val;
                                }
                            }
                        }
                    } else {
                        // I4x4 via CABAC: existing path
                        for blk in 0..16 {
                            let (blk_row, blk_col) = BLOCK_INDEX_TO_OFFSET[blk];
                            let px = mb_x + blk_col;
                            let py = mb_y + blk_row;

                            let mut block_coeffs = [0i32; 16];
                            if cbp_luma & (1 << (blk / 4)) != 0 {
                                let left_nz_blk = cabac_neighbor_nz_luma(
                                    &nc_luma,
                                    mb_idx,
                                    mb_width as usize,
                                    blk,
                                    true,
                                    true,
                                    &mb_slice_id,
                                    this_slice_id,
                                );
                                let top_nz_blk = cabac_neighbor_nz_luma(
                                    &nc_luma,
                                    mb_idx,
                                    mb_width as usize,
                                    blk,
                                    false,
                                    true,
                                    &mb_slice_id,
                                    this_slice_id,
                                );

                                let cbf =
                                    cr.decode_coded_block_flag(st, 2, left_nz_blk, top_nz_blk);
                                if cbf {
                                    let (coeffs, tc) = cr.decode_residual_cabac(st, 2, 16);
                                    nc_luma[mb_idx * 16 + blk] = tc;
                                    for (pos, val) in &coeffs {
                                        let (r, c) = ZIGZAG_4X4[*pos];
                                        block_coeffs[r * 4 + c] = *val;
                                    }
                                    dequant_4x4_full(
                                        &mut block_coeffs,
                                        qp_y,
                                        &pps.scaling_list_4x4[0],
                                    );
                                }
                            }
                            inverse_dct_4x4(&mut block_coeffs);

                            // I4x4 prediction
                            let local_row = py - mb_y;
                            let local_col = px - mb_x;
                            // Above samples: unavailable if at picture top or cross-slice MB boundary
                            let above_avail = py > 0 && (local_row > 0 || above_mb_avail);
                            let above_buf: Option<[u8; 8]> = if above_avail {
                                let mut buf = [0u8; 8];
                                for (i, b) in buf.iter_mut().enumerate().take(4) {
                                    *b = frame.y[(py - 1) * stride + px + i];
                                }
                                let topright_avail = if local_row == 0 {
                                    if px + 4 < (mb_x + 16).min(stride) {
                                        true // within same MB
                                    } else if px + 4 < stride {
                                        above_right_mb_avail
                                    } else {
                                        false
                                    }
                                } else {
                                    !matches!(blk, 3 | 7 | 11 | 13 | 15)
                                };
                                if topright_avail {
                                    for (i, b) in buf.iter_mut().enumerate().skip(4) {
                                        let col = (px + i).min(stride - 1);
                                        *b = frame.y[(py - 1) * stride + col];
                                    }
                                } else {
                                    let last = buf[3];
                                    buf[4..8].fill(last);
                                }
                                Some(buf)
                            } else {
                                None
                            };
                            // Left samples: unavailable if at picture left or cross-slice MB boundary
                            let left_avail = px > 0 && (local_col > 0 || left_mb_avail);
                            let left_buf: Option<[u8; 4]> = if left_avail {
                                let mut buf = [0u8; 4];
                                for (i, b) in buf.iter_mut().enumerate() {
                                    *b = frame.y[(py + i) * stride + px - 1];
                                }
                                Some(buf)
                            } else {
                                None
                            };
                            // Above-left: unavailable if cross-slice
                            let al_avail = px > 0
                                && py > 0
                                && ((local_row > 0 && local_col > 0)
                                    || (local_row > 0 && local_col == 0 && left_mb_avail)
                                    || (local_row == 0 && local_col > 0 && above_mb_avail)
                                    || (local_row == 0 && local_col == 0 && above_left_mb_avail));
                            let above_left_val = if al_avail {
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
                            for r in 0..4 {
                                for c in 0..4 {
                                    let val = (pred[r * 4 + c] as i32 + block_coeffs[r * 4 + c])
                                        .clamp(0, 255)
                                        as u8;
                                    frame.y[(py + r) * stride + px + c] = val;
                                }
                            }
                        }
                    } // close if use_8x8_intra else (luma only)

                    // Chroma (simplified — reuse existing chroma decode pattern)
                    let chroma_width = (width / 2) as usize;
                    let chroma_mb_x = mb_x / 2;
                    let chroma_mb_y = mb_y / 2;

                    // Chroma prediction (slice boundary: cross-slice neighbors unavailable)
                    let (pred_u, pred_v) = make_ctx!().predict_chroma_intra(
                        mb_x,
                        mb_y,
                        intra_chroma_pred_mode,
                        above_mb_avail,
                        left_mb_avail,
                        above_left_mb_avail,
                    );

                    // Chroma residual: DC + AC via CABAC
                    let mut chroma_dc_cb = [0i32; 4];
                    let mut chroma_dc_cr = [0i32; 4];
                    if cbp_chroma >= 1 {
                        // Chroma DC CBF context: uses bits 6-7 of neighbor cbp_table
                        let left_dc_nz = if !mb_idx.is_multiple_of(mb_width as usize)
                            && mb_slice_id[mb_idx - 1] == this_slice_id
                        {
                            (mb_cbp[mb_idx - 1] >> 6) & 1 != 0
                        } else {
                            true
                        }; // unavailable intra: 0x7CF bit 6 = 1
                        let top_dc_nz = if mb_idx >= mb_width as usize
                            && mb_slice_id[mb_idx - mb_width as usize] == this_slice_id
                        {
                            (mb_cbp[mb_idx - mb_width as usize] >> 6) & 1 != 0
                        } else {
                            true
                        };
                        if cr.decode_coded_block_flag(st, 3, left_dc_nz, top_dc_nz) {
                            let (coeffs, _tc) = cr.decode_residual_cabac(st, 3, 4);
                            for (pos, val) in coeffs {
                                chroma_dc_cb[pos] = val;
                            }
                            mb_cbp[mb_idx] |= 0x40; // set Cb DC coded flag
                        }
                        let left_dc_nz_cr = if !mb_idx.is_multiple_of(mb_width as usize)
                            && mb_slice_id[mb_idx - 1] == this_slice_id
                        {
                            (mb_cbp[mb_idx - 1] >> 7) & 1 != 0
                        } else {
                            true
                        };
                        let top_dc_nz_cr = if mb_idx >= mb_width as usize
                            && mb_slice_id[mb_idx - mb_width as usize] == this_slice_id
                        {
                            (mb_cbp[mb_idx - mb_width as usize] >> 7) & 1 != 0
                        } else {
                            true
                        };
                        if cr.decode_coded_block_flag(st, 3, left_dc_nz_cr, top_dc_nz_cr) {
                            let (coeffs, _tc) = cr.decode_residual_cabac(st, 3, 4);
                            for (pos, val) in coeffs {
                                chroma_dc_cr[pos] = val;
                            }
                            mb_cbp[mb_idx] |= 0x80; // set Cr DC coded flag
                        }
                    }
                    let mut chroma_ac_cb = [[0i32; 15]; 4];
                    let mut chroma_ac_cr = [[0i32; 15]; 4];
                    if cbp_chroma >= 2 {
                        // Chroma AC: cat=4, max_coeff=15
                        for blk in 0..4 {
                            let left_nz = cabac_neighbor_nz_chroma(
                                &nc_cb,
                                mb_idx,
                                mb_width as usize,
                                blk,
                                true,
                                true,
                                &mb_slice_id,
                                this_slice_id,
                            );
                            let top_nz = cabac_neighbor_nz_chroma(
                                &nc_cb,
                                mb_idx,
                                mb_width as usize,
                                blk,
                                false,
                                true,
                                &mb_slice_id,
                                this_slice_id,
                            );
                            if cr.decode_coded_block_flag(st, 4, left_nz, top_nz) {
                                let (coeffs, tc) = cr.decode_residual_cabac(st, 4, 15);
                                nc_cb[mb_idx * 4 + blk] = tc;
                                for (pos, val) in coeffs {
                                    chroma_ac_cb[blk][pos] = val;
                                }
                            }
                        }
                        for blk in 0..4 {
                            let left_nz = cabac_neighbor_nz_chroma(
                                &nc_cr,
                                mb_idx,
                                mb_width as usize,
                                blk,
                                true,
                                true,
                                &mb_slice_id,
                                this_slice_id,
                            );
                            let top_nz = cabac_neighbor_nz_chroma(
                                &nc_cr,
                                mb_idx,
                                mb_width as usize,
                                blk,
                                false,
                                true,
                                &mb_slice_id,
                                this_slice_id,
                            );
                            if cr.decode_coded_block_flag(st, 4, left_nz, top_nz) {
                                let (coeffs, tc) = cr.decode_residual_cabac(st, 4, 15);
                                nc_cr[mb_idx * 4 + blk] = tc;
                                for (pos, val) in coeffs {
                                    chroma_ac_cr[blk][pos] = val;
                                }
                            }
                        }
                    }

                    // Reconstruct chroma for each plane
                    for (plane_dc, plane_ac, pred_plane, frame_plane, scale_idx) in [
                        (
                            &mut chroma_dc_cb,
                            &chroma_ac_cb,
                            &pred_u,
                            &mut frame.u,
                            1usize,
                        ),
                        (
                            &mut chroma_dc_cr,
                            &chroma_ac_cr,
                            &pred_v,
                            &mut frame.v,
                            2usize,
                        ),
                    ] {
                        let chroma_scale = &pps.scaling_list_4x4[scale_idx];
                        if cbp_chroma >= 1 {
                            inverse_hadamard_2x2(plane_dc);
                            dequant_chroma_dc(plane_dc, _qp_c, chroma_scale[0]);
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
                                dequant_4x4_ac_raster(&mut block_raster, _qp_c, chroma_scale);
                            }
                            inverse_dct_4x4(&mut block_raster);
                            for r in 0..4 {
                                for c in 0..4 {
                                    chroma_residual[(blk_row + r) * 8 + blk_col + c] =
                                        block_raster[r * 4 + c];
                                }
                            }
                        }

                        for y in 0..8 {
                            for x in 0..8 {
                                let val = (pred_plane[y * 8 + x] as i32
                                    + chroma_residual[y * 8 + x])
                                    .clamp(0, 255) as u8;
                                frame_plane[(chroma_mb_y + y) * chroma_width + chroma_mb_x + x] =
                                    val;
                            }
                        }
                    }

                    mb_info[mb_idx] = MbInfo {
                        mb_type: MbType::Intra,
                        qp_y,
                        ..Default::default()
                    };
                    // I4x4/I8x8 is NOT I16x16 for CABAC context
                } else if mb_type <= 24 {
                    // I16x16 via CABAC
                    is_i16x16[mb_idx] = true;
                    let mt = mb_type - 1;
                    let intra16x16_pred_mode = (mt % 4) as u8;
                    let cbp_chroma = ((mt / 4) % 3) as u8;
                    let cbp_luma = if mt >= 12 { 15u8 } else { 0u8 };

                    let left_cm = if !mb_idx.is_multiple_of(mb_width as usize)
                        && mb_slice_id[mb_idx - 1] == this_slice_id
                    {
                        mb_chroma_pred[mb_idx - 1]
                    } else {
                        0
                    };
                    let top_cm = if mb_idx >= mb_width as usize
                        && mb_slice_id[mb_idx - mb_width as usize] == this_slice_id
                    {
                        mb_chroma_pred[mb_idx - mb_width as usize]
                    } else {
                        0
                    };
                    let intra_chroma_pred_mode = cr.decode_chroma_pred_mode(st, left_cm, top_cm);
                    mb_chroma_pred[mb_idx] = intra_chroma_pred_mode;

                    let delta = cr.decode_mb_qp_delta(st, last_qp_delta_nonzero);
                    last_qp_delta_nonzero = delta != 0;
                    let qp_y = ((prev_mb_qp + delta + 52) % 52 + 52) % 52;
                    prev_mb_qp = qp_y;
                    let qp_c = chroma_qp(qp_y, pps.chroma_qp_index_offset);

                    // Luma DC: cat=0, 16 coefficients
                    // CBF context uses bit 8 of neighbor cbp_table (luma DC coded flag)
                    let mut luma_dc = [0i32; 16];
                    let dc_left_nz = if !mb_idx.is_multiple_of(mb_width as usize)
                        && mb_slice_id[mb_idx - 1] == this_slice_id
                    {
                        (mb_cbp[mb_idx - 1] >> 8) & 1 != 0
                    } else {
                        true
                    }; // unavailable intra: 0x7CF bit 8 = 1 (0x7CF = 0b0111_1100_1111)
                    let dc_top_nz = if mb_idx >= mb_width as usize
                        && mb_slice_id[mb_idx - mb_width as usize] == this_slice_id
                    {
                        (mb_cbp[mb_idx - mb_width as usize] >> 8) & 1 != 0
                    } else {
                        true
                    };
                    if cr.decode_coded_block_flag(st, 0, dc_left_nz, dc_top_nz) {
                        let (coeffs, _tc) = cr.decode_residual_cabac(st, 0, 16);
                        for (pos, val) in coeffs {
                            luma_dc[pos] = val;
                        }
                        mb_cbp[mb_idx] |= 0x100; // set luma DC coded flag
                    }

                    // Luma AC: cat=1, 15 coefficients per block
                    let mut luma_ac_scan = [[0i32; 15]; 16];
                    if cbp_luma != 0 {
                        for blk in 0..16 {
                            let left_nz = cabac_neighbor_nz_luma(
                                &nc_luma,
                                mb_idx,
                                mb_width as usize,
                                blk,
                                true,
                                true,
                                &mb_slice_id,
                                this_slice_id,
                            );
                            let top_nz = cabac_neighbor_nz_luma(
                                &nc_luma,
                                mb_idx,
                                mb_width as usize,
                                blk,
                                false,
                                true,
                                &mb_slice_id,
                                this_slice_id,
                            );
                            if cr.decode_coded_block_flag(st, 1, left_nz, top_nz) {
                                let (coeffs, tc) = cr.decode_residual_cabac(st, 1, 15);
                                nc_luma[mb_idx * 16 + blk] = tc;
                                for (pos, val) in coeffs {
                                    luma_ac_scan[blk][pos] = val;
                                }
                            }
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

                    const DC_RASTER_TO_BLOCK: [usize; 16] =
                        [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15];

                    let mut luma_residual = [0i32; 256];
                    for blk in 0..16 {
                        let (blk_row, blk_col) = BLOCK_INDEX_TO_OFFSET[blk];
                        let mut block_raster = [0i32; 16];
                        let dc_idx = DC_RASTER_TO_BLOCK.iter().position(|&b| b == blk).unwrap();
                        block_raster[0] = luma_dc_raster[dc_idx];

                        if cbp_luma != 0 {
                            for scan_idx in 0..15 {
                                let (r, c) = ZIGZAG_4X4[scan_idx + 1];
                                block_raster[r * 4 + c] = luma_ac_scan[blk][scan_idx];
                            }
                            dequant_4x4_ac_raster(
                                &mut block_raster,
                                qp_y,
                                &pps.scaling_list_4x4[0],
                            );
                        }

                        inverse_dct_4x4(&mut block_raster);

                        for r in 0..4 {
                            for c in 0..4 {
                                luma_residual[(blk_row + r) * 16 + blk_col + c] =
                                    block_raster[r * 4 + c];
                            }
                        }
                    }

                    // I16x16 prediction + residual
                    make_ctx!().reconstruct_luma_16x16(
                        mb_x,
                        mb_y,
                        intra16x16_pred_mode,
                        &luma_residual,
                        above_mb_avail_i,
                        left_mb_avail_i,
                        above_left_mb_avail_i,
                    );

                    // Chroma (same as I4x4 CABAC path)
                    let chroma_width = (width / 2) as usize;
                    let chroma_mb_x = mb_x / 2;
                    let chroma_mb_y = mb_y / 2;
                    let (pred_u, pred_v) = make_ctx!().predict_chroma_intra(
                        mb_x,
                        mb_y,
                        intra_chroma_pred_mode,
                        above_mb_avail_i,
                        left_mb_avail_i,
                        above_left_mb_avail_i,
                    );

                    // Chroma residual
                    let mut chroma_dc_cb = [0i32; 4];
                    let mut chroma_dc_cr = [0i32; 4];
                    if cbp_chroma >= 1 {
                        let left_dc_nz = if !mb_idx.is_multiple_of(mb_width as usize)
                            && mb_slice_id[mb_idx - 1] == this_slice_id
                        {
                            (mb_cbp[mb_idx - 1] >> 6) & 1 != 0
                        } else {
                            true
                        };
                        let top_dc_nz = if mb_idx >= mb_width as usize
                            && mb_slice_id[mb_idx - mb_width as usize] == this_slice_id
                        {
                            (mb_cbp[mb_idx - mb_width as usize] >> 6) & 1 != 0
                        } else {
                            true
                        };
                        if cr.decode_coded_block_flag(st, 3, left_dc_nz, top_dc_nz) {
                            let (coeffs, _tc) = cr.decode_residual_cabac(st, 3, 4);
                            for (pos, val) in coeffs {
                                chroma_dc_cb[pos] = val;
                            }
                            mb_cbp[mb_idx] |= 0x40;
                        }
                        let left_dc_nz_cr = if !mb_idx.is_multiple_of(mb_width as usize)
                            && mb_slice_id[mb_idx - 1] == this_slice_id
                        {
                            (mb_cbp[mb_idx - 1] >> 7) & 1 != 0
                        } else {
                            true
                        };
                        let top_dc_nz_cr = if mb_idx >= mb_width as usize
                            && mb_slice_id[mb_idx - mb_width as usize] == this_slice_id
                        {
                            (mb_cbp[mb_idx - mb_width as usize] >> 7) & 1 != 0
                        } else {
                            true
                        };
                        if cr.decode_coded_block_flag(st, 3, left_dc_nz_cr, top_dc_nz_cr) {
                            let (coeffs, _tc) = cr.decode_residual_cabac(st, 3, 4);
                            for (pos, val) in coeffs {
                                chroma_dc_cr[pos] = val;
                            }
                            mb_cbp[mb_idx] |= 0x80;
                        }
                    }
                    let mut chroma_ac_cb = [[0i32; 15]; 4];
                    let mut chroma_ac_cr = [[0i32; 15]; 4];
                    if cbp_chroma >= 2 {
                        for blk in 0..4 {
                            let left_nz = cabac_neighbor_nz_chroma(
                                &nc_cb,
                                mb_idx,
                                mb_width as usize,
                                blk,
                                true,
                                true,
                                &mb_slice_id,
                                this_slice_id,
                            );
                            let top_nz = cabac_neighbor_nz_chroma(
                                &nc_cb,
                                mb_idx,
                                mb_width as usize,
                                blk,
                                false,
                                true,
                                &mb_slice_id,
                                this_slice_id,
                            );
                            if cr.decode_coded_block_flag(st, 4, left_nz, top_nz) {
                                let (coeffs, tc) = cr.decode_residual_cabac(st, 4, 15);
                                nc_cb[mb_idx * 4 + blk] = tc;
                                for (pos, val) in coeffs {
                                    chroma_ac_cb[blk][pos] = val;
                                }
                            }
                        }
                        for blk in 0..4 {
                            let left_nz = cabac_neighbor_nz_chroma(
                                &nc_cr,
                                mb_idx,
                                mb_width as usize,
                                blk,
                                true,
                                true,
                                &mb_slice_id,
                                this_slice_id,
                            );
                            let top_nz = cabac_neighbor_nz_chroma(
                                &nc_cr,
                                mb_idx,
                                mb_width as usize,
                                blk,
                                false,
                                true,
                                &mb_slice_id,
                                this_slice_id,
                            );
                            if cr.decode_coded_block_flag(st, 4, left_nz, top_nz) {
                                let (coeffs, tc) = cr.decode_residual_cabac(st, 4, 15);
                                nc_cr[mb_idx * 4 + blk] = tc;
                                for (pos, val) in coeffs {
                                    chroma_ac_cr[blk][pos] = val;
                                }
                            }
                        }
                    }

                    for (plane_dc, plane_ac, pred_plane, frame_plane, scale_idx) in [
                        (
                            &mut chroma_dc_cb,
                            &chroma_ac_cb,
                            &pred_u,
                            &mut frame.u,
                            1usize,
                        ),
                        (
                            &mut chroma_dc_cr,
                            &chroma_ac_cr,
                            &pred_v,
                            &mut frame.v,
                            2usize,
                        ),
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
                        for y in 0..8 {
                            for x in 0..8 {
                                let val = (pred_plane[y * 8 + x] as i32
                                    + chroma_residual[y * 8 + x])
                                    .clamp(0, 255) as u8;
                                frame_plane[(chroma_mb_y + y) * chroma_width + chroma_mb_x + x] =
                                    val;
                            }
                        }
                    }

                    mb_info[mb_idx] = MbInfo {
                        mb_type: MbType::Intra,
                        qp_y,
                        ..Default::default()
                    };
                    // CBP already partially set during DC/AC decode (bits 6-7)
                    mb_cbp[mb_idx] |= (cbp_luma as u16) | ((cbp_chroma as u16) << 4);
                } else {
                    // I_PCM via CABAC (mb_type == 25)
                    let pcm_pos = cr.pcm_byte_position();
                    let pcm_data = &nal.rbsp[pcm_pos..];
                    let mut off = 0;
                    for r in 0..16 {
                        for c in 0..16 {
                            frame.y[(mb_y + r) * stride + mb_x + c] = pcm_data[off];
                            off += 1;
                        }
                    }
                    let cw = (width / 2) as usize;
                    let cx = mb_x / 2;
                    let cy = mb_y / 2;
                    for r in 0..8 {
                        for c in 0..8 {
                            frame.u[(cy + r) * cw + cx + c] = pcm_data[off];
                            off += 1;
                        }
                    }
                    for r in 0..8 {
                        for c in 0..8 {
                            frame.v[(cy + r) * cw + cx + c] = pcm_data[off];
                            off += 1;
                        }
                    }
                    for blk in 0..16 {
                        nc_luma[mb_idx * 16 + blk] = 16;
                    }
                    for blk in 0..4 {
                        nc_cb[mb_idx * 4 + blk] = 16;
                        nc_cr[mb_idx * 4 + blk] = 16;
                    }
                    cr.reinit(pcm_pos + off);
                    prev_mb_qp = 0;
                    last_qp_delta_nonzero = false;
                    is_i16x16[mb_idx] = false;
                    mb_cbp[mb_idx] = 0;
                    mb_info[mb_idx] = MbInfo {
                        mb_type: MbType::Ipcm,
                        qp_y: 0,
                        ..Default::default()
                    };
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
                        make_ctx!().decode_p_skip_mb(
                            mb_idx,
                            mb_x,
                            mb_y,
                            &ref_pic_list,
                            &wctx,
                            use_weight,
                        );
                    } else {
                        // B_Skip: spatial/temporal direct MV + MC, no residual
                        make_ctx!().decode_b_skip_mb(
                            mb_idx,
                            mb_x,
                            mb_y,
                            header.direct_spatial_mv_pred_flag,
                            sps.direct_8x8_inference_flag,
                            current_poc,
                            &_ref_pic_list_l0,
                            &_ref_pic_list_l1,
                            &wctx,
                            use_weight,
                        );
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

            let raw_mb_type = reader.read_ue()?;
            // For P slices, mb_type >= 5 means intra (subtract 5)
            // For B slices, mb_type >= 23 means intra (subtract 23)
            let inter_limit = if is_p_slice {
                5
            } else if is_b_slice {
                23
            } else {
                0
            };
            let (mb_type, is_inter) = if (is_p_slice || is_b_slice) && raw_mb_type < inter_limit {
                (raw_mb_type, true)
            } else {
                (raw_mb_type - inter_limit, false)
            };

            if is_inter && is_b_slice {
                // === Inter (B) macroblock ===
                // Table 7-11: mb_type 0=B_Direct_16x16, 1=B_L0_16x16,
                // 2=B_L1_16x16, 3=B_Bi_16x16, 4-21=16x8/8x16 variants, 22=B_8x8
                let mut no_sub_less_8x8_b = true; // for transform_size_8x8_flag
                struct SubPart {
                    x: usize,
                    y: usize,
                    w: usize,
                    h: usize,
                    ref_idx_l0: i8,
                    ref_idx_l1: i8,
                    mv_l0: [i16; 2],
                    mv_l1: [i16; 2],
                    pred_l0: bool,
                    pred_l1: bool,
                }
                let mut sub_parts: Vec<SubPart> = Vec::new();

                match mb_type {
                    1 | 2 => {
                        // B_L0_16x16 (mb_type=1) or B_L1_16x16 (mb_type=2)
                        let is_l0 = mb_type == 1;
                        let num_active = if is_l0 {
                            header.num_ref_idx_l0_active
                        } else {
                            header.num_ref_idx_l1_active
                        };

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
                            mv_store_ref,
                            ref_store_ref,
                            mb_idx,
                            mb_width as usize,
                            0,
                            16,
                            16,
                            ref_idx,
                            &mb_slice_id,
                            this_slice_id,
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
                            x: 0,
                            y: 0,
                            w: 16,
                            h: 16,
                            ref_idx_l0: if is_l0 { ref_idx } else { -1 },
                            ref_idx_l1: if is_l0 { -1 } else { ref_idx },
                            mv_l0: if is_l0 { mv } else { [0, 0] },
                            mv_l1: if is_l0 { [0, 0] } else { mv },
                            pred_l0: is_l0,
                            pred_l1: !is_l0,
                        });
                    }
                    0 => {
                        // B_Direct_16x16: derive MVs per 4x4 block via spatial or temporal direct
                        if !sps.direct_8x8_inference_flag {
                            no_sub_less_8x8_b = false;
                        }
                        make_ctx!().derive_direct_mvs(
                            mb_idx,
                            0,
                            16,
                            header.direct_spatial_mv_pred_flag,
                            sps.direct_8x8_inference_flag,
                            current_poc,
                            &_ref_pic_list_l0,
                            &_ref_pic_list_l1,
                        );
                        // Build sub_parts for MC, coalescing blocks with identical MVs/refs
                        // per 8x8 sub-block (spec 8.4.1.2.1: direct mode applied per 8x8)
                        let base = mb_idx * 16;
                        for i8x8 in 0..4 {
                            let blk0 = i8x8 * 4; // top-left 4x4 block of this 8x8
                            let mv0 = mv_store_l0[base + blk0];
                            let mv1 = mv_store_l1[base + blk0];
                            let r0 = ref_idx_store_l0[base + blk0];
                            let r1 = ref_idx_store_l1[base + blk0];
                            // Check if all 4 blocks in this 8x8 have the same MV/ref
                            let uniform = (1..4).all(|sub| {
                                let b = blk0 + sub;
                                mv_store_l0[base + b] == mv0
                                    && mv_store_l1[base + b] == mv1
                                    && ref_idx_store_l0[base + b] == r0
                                    && ref_idx_store_l1[base + b] == r1
                            });
                            let (blk_row, blk_col) = BLOCK_INDEX_TO_OFFSET[blk0];
                            if uniform {
                                sub_parts.push(SubPart {
                                    x: blk_col,
                                    y: blk_row,
                                    w: 8,
                                    h: 8,
                                    ref_idx_l0: r0,
                                    ref_idx_l1: r1,
                                    mv_l0: mv0,
                                    mv_l1: mv1,
                                    pred_l0: r0 >= 0,
                                    pred_l1: r1 >= 0,
                                });
                            } else {
                                for sub in 0..4 {
                                    let b = blk0 + sub;
                                    let (br, bc) = BLOCK_INDEX_TO_OFFSET[b];
                                    let rl0 = ref_idx_store_l0[base + b];
                                    let rl1 = ref_idx_store_l1[base + b];
                                    sub_parts.push(SubPart {
                                        x: bc,
                                        y: br,
                                        w: 4,
                                        h: 4,
                                        ref_idx_l0: rl0,
                                        ref_idx_l1: rl1,
                                        mv_l0: mv_store_l0[base + b],
                                        mv_l1: mv_store_l1[base + b],
                                        pred_l0: rl0 >= 0,
                                        pred_l1: rl1 >= 0,
                                    });
                                }
                            }
                        }
                    }
                    3 => {
                        // B_Bi_16x16: both L0 and L1, averaged
                        let ref_idx_l0 = if header.num_ref_idx_l0_active > 1 {
                            reader.read_te(header.num_ref_idx_l0_active - 1)? as i8
                        } else {
                            0
                        };
                        let ref_idx_l1 = if header.num_ref_idx_l1_active > 1 {
                            reader.read_te(header.num_ref_idx_l1_active - 1)? as i8
                        } else {
                            0
                        };

                        let mvd_l0_x = reader.read_se()? as i16;
                        let mvd_l0_y = reader.read_se()? as i16;
                        let (mvp_l0_x, mvp_l0_y) = predict_mv(
                            &mv_store_l0,
                            &ref_idx_store_l0,
                            mb_idx,
                            mb_width as usize,
                            0,
                            16,
                            16,
                            ref_idx_l0,
                            &mb_slice_id,
                            this_slice_id,
                        );
                        let mv_l0 = [mvp_l0_x + mvd_l0_x, mvp_l0_y + mvd_l0_y];

                        let mvd_l1_x = reader.read_se()? as i16;
                        let mvd_l1_y = reader.read_se()? as i16;
                        let (mvp_l1_x, mvp_l1_y) = predict_mv(
                            &mv_store_l1,
                            &ref_idx_store_l1,
                            mb_idx,
                            mb_width as usize,
                            0,
                            16,
                            16,
                            ref_idx_l1,
                            &mb_slice_id,
                            this_slice_id,
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
                            x: 0,
                            y: 0,
                            w: 16,
                            h: 16,
                            ref_idx_l0,
                            ref_idx_l1,
                            mv_l0,
                            mv_l1,
                            pred_l0: true,
                            pred_l1: true,
                        });
                    }
                    4..=21 => {
                        // B 16x8/8x16 variants (Table 7-11)
                        // Each type specifies partition size and per-partition pred direction
                        // Format: (part_w, part_h, [(pred_l0_0, pred_l1_0), (pred_l0_1, pred_l1_1)])
                        #[rustfmt::skip]
                        #[allow(clippy::type_complexity)]
                        const B_PART_TABLE: [(usize, usize, [(bool, bool); 2]); 18] = [
                            // mb_type 4-5: B_L0_L0
                            (16, 8, [(true,false),(true,false)]),  // 4
                            (8, 16, [(true,false),(true,false)]),  // 5
                            // mb_type 6-7: B_L1_L1
                            (16, 8, [(false,true),(false,true)]), // 6
                            (8, 16, [(false,true),(false,true)]), // 7
                            // mb_type 8-9: B_L0_L1
                            (16, 8, [(true,false),(false,true)]), // 8
                            (8, 16, [(true,false),(false,true)]), // 9
                            // mb_type 10-11: B_L1_L0
                            (16, 8, [(false,true),(true,false)]), // 10
                            (8, 16, [(false,true),(true,false)]), // 11
                            // mb_type 12-13: B_L0_Bi
                            (16, 8, [(true,false),(true,true)]),  // 12
                            (8, 16, [(true,false),(true,true)]),  // 13
                            // mb_type 14-15: B_L1_Bi
                            (16, 8, [(false,true),(true,true)]),  // 14
                            (8, 16, [(false,true),(true,true)]),  // 15
                            // mb_type 16-17: B_Bi_L0
                            (16, 8, [(true,true),(true,false)]),  // 16
                            (8, 16, [(true,true),(true,false)]),  // 17
                            // mb_type 18-19: B_Bi_L1
                            (16, 8, [(true,true),(false,true)]),  // 18
                            (8, 16, [(true,true),(false,true)]),  // 19
                            // mb_type 20-21: B_Bi_Bi
                            (16, 8, [(true,true),(true,true)]),   // 20
                            (8, 16, [(true,true),(true,true)]),   // 21
                        ];
                        let entry = B_PART_TABLE[(mb_type - 4) as usize];
                        let (part_w, part_h) = (entry.0, entry.1);
                        let pred_flags = entry.2;

                        // Parse ref_idx: for each list, for each partition (spec 7.3.5.1)
                        let mut part_ref_l0 = [-1i8; 2];
                        let mut part_ref_l1 = [-1i8; 2];
                        for p in 0..2 {
                            if pred_flags[p].0 && header.num_ref_idx_l0_active > 1 {
                                part_ref_l0[p] =
                                    reader.read_te(header.num_ref_idx_l0_active - 1)? as i8;
                            } else if pred_flags[p].0 {
                                part_ref_l0[p] = 0;
                            }
                        }
                        for p in 0..2 {
                            if pred_flags[p].1 && header.num_ref_idx_l1_active > 1 {
                                part_ref_l1[p] =
                                    reader.read_te(header.num_ref_idx_l1_active - 1)? as i8;
                            } else if pred_flags[p].1 {
                                part_ref_l1[p] = 0;
                            }
                        }

                        // Parse MVD: for each list, for each partition (spec 7.3.5.1)
                        // Order: L0 part0, L0 part1, L1 part0, L1 part1
                        let mut mv_l0_parts = [[0i16; 2]; 2];
                        let mut mv_l1_parts = [[0i16; 2]; 2];
                        for p in 0..2 {
                            if pred_flags[p].0 {
                                // Store ref_idx_l0 before predicting (partition 1 needs partition 0's data)
                                let (py_off, px_off) =
                                    if part_h == 8 { (p * 8, 0) } else { (0, p * 8) };
                                for r in (0..part_h).step_by(4) {
                                    for c in (0..part_w).step_by(4) {
                                        let lr = (py_off + r) / 4;
                                        let lc = (px_off + c) / 4;
                                        if let Some(blk) = BLOCK_INDEX_TO_OFFSET
                                            .iter()
                                            .position(|&(br, bc)| br / 4 == lr && bc / 4 == lc)
                                        {
                                            ref_idx_store_l0[mb_idx * 16 + blk] = part_ref_l0[p];
                                        }
                                    }
                                }
                                let mvd_x = reader.read_se()? as i16;
                                let mvd_y = reader.read_se()? as i16;
                                let (mvp_x, mvp_y) = predict_mv(
                                    &mv_store_l0,
                                    &ref_idx_store_l0,
                                    mb_idx,
                                    mb_width as usize,
                                    p,
                                    part_w,
                                    part_h,
                                    part_ref_l0[p],
                                    &mb_slice_id,
                                    this_slice_id,
                                );
                                mv_l0_parts[p] = [mvp_x + mvd_x, mvp_y + mvd_y];
                                // Store MV immediately for partition 1 to read partition 0
                                for r in (0..part_h).step_by(4) {
                                    for c in (0..part_w).step_by(4) {
                                        let lr = (py_off + r) / 4;
                                        let lc = (px_off + c) / 4;
                                        if let Some(blk) = BLOCK_INDEX_TO_OFFSET
                                            .iter()
                                            .position(|&(br, bc)| br / 4 == lr && bc / 4 == lc)
                                        {
                                            mv_store_l0[mb_idx * 16 + blk] = mv_l0_parts[p];
                                        }
                                    }
                                }
                            }
                        }
                        for p in 0..2 {
                            if pred_flags[p].1 {
                                let (py_off, px_off) =
                                    if part_h == 8 { (p * 8, 0) } else { (0, p * 8) };
                                for r in (0..part_h).step_by(4) {
                                    for c in (0..part_w).step_by(4) {
                                        let lr = (py_off + r) / 4;
                                        let lc = (px_off + c) / 4;
                                        if let Some(blk) = BLOCK_INDEX_TO_OFFSET
                                            .iter()
                                            .position(|&(br, bc)| br / 4 == lr && bc / 4 == lc)
                                        {
                                            ref_idx_store_l1[mb_idx * 16 + blk] = part_ref_l1[p];
                                        }
                                    }
                                }
                                let mvd_x = reader.read_se()? as i16;
                                let mvd_y = reader.read_se()? as i16;
                                let (mvp_x, mvp_y) = predict_mv(
                                    &mv_store_l1,
                                    &ref_idx_store_l1,
                                    mb_idx,
                                    mb_width as usize,
                                    p,
                                    part_w,
                                    part_h,
                                    part_ref_l1[p],
                                    &mb_slice_id,
                                    this_slice_id,
                                );
                                mv_l1_parts[p] = [mvp_x + mvd_x, mvp_y + mvd_y];
                                for r in (0..part_h).step_by(4) {
                                    for c in (0..part_w).step_by(4) {
                                        let lr = (py_off + r) / 4;
                                        let lc = (px_off + c) / 4;
                                        if let Some(blk) = BLOCK_INDEX_TO_OFFSET
                                            .iter()
                                            .position(|&(br, bc)| br / 4 == lr && bc / 4 == lc)
                                        {
                                            mv_store_l1[mb_idx * 16 + blk] = mv_l1_parts[p];
                                        }
                                    }
                                }
                            }
                        }

                        // Build sub_parts for MC
                        for p in 0..2 {
                            let (py_off, px_off) =
                                if part_h == 8 { (p * 8, 0) } else { (0, p * 8) };
                            sub_parts.push(SubPart {
                                x: px_off,
                                y: py_off,
                                w: part_w,
                                h: part_h,
                                ref_idx_l0: part_ref_l0[p],
                                ref_idx_l1: part_ref_l1[p],
                                mv_l0: mv_l0_parts[p],
                                mv_l1: mv_l1_parts[p],
                                pred_l0: pred_flags[p].0,
                                pred_l1: pred_flags[p].1,
                            });
                        }
                    }
                    22 => {
                        // B_8x8: 4 sub-MBs, each with its own sub_mb_type
                        // Table 7-17: sub_mb_type 0-12
                        // Format: (sub_w, sub_h, pred_l0, pred_l1)
                        #[rustfmt::skip]
                        const B_SUB_TABLE: [(usize, usize, bool, bool); 13] = [
                            (8, 8, false, false), // 0: B_Direct_8x8
                            (8, 8, true, false),  // 1: B_L0_8x8
                            (8, 8, false, true),  // 2: B_L1_8x8
                            (8, 8, true, true),   // 3: B_Bi_8x8
                            (8, 4, true, false),  // 4: B_L0_8x4
                            (4, 8, true, false),  // 5: B_L0_4x8
                            (8, 4, false, true),  // 6: B_L1_8x4
                            (4, 8, false, true),  // 7: B_L1_4x8
                            (8, 4, true, true),   // 8: B_Bi_8x4
                            (4, 8, true, true),   // 9: B_Bi_4x8
                            (4, 4, true, false),  // 10: B_L0_4x4
                            (4, 4, false, true),  // 11: B_L1_4x4
                            (4, 4, true, true),   // 12: B_Bi_4x4
                        ];

                        let sub_mb_origins = [(0, 0), (0, 8), (8, 0), (8, 8)];
                        let mut sub_mb_types = [0u32; 4];
                        for smt in &mut sub_mb_types {
                            *smt = reader.read_ue()?;
                        }
                        if sub_mb_types
                            .iter()
                            .any(|&smt| smt > 3 || (smt == 0 && !sps.direct_8x8_inference_flag))
                        {
                            no_sub_less_8x8_b = false;
                        }

                        // Parse ref_idx for each 8x8 sub-MB
                        let mut sub_ref_l0 = [-1i8; 4];
                        let mut sub_ref_l1 = [-1i8; 4];
                        for smb in 0..4 {
                            if sub_mb_types[smb] == 0 {
                                continue;
                            } // B_Direct_8x8: no ref_idx
                            let (_, _, pl0, _) = B_SUB_TABLE[sub_mb_types[smb] as usize];
                            if pl0 && header.num_ref_idx_l0_active > 1 {
                                sub_ref_l0[smb] =
                                    reader.read_te(header.num_ref_idx_l0_active - 1)? as i8;
                            } else if pl0 {
                                sub_ref_l0[smb] = 0;
                            }
                        }
                        for smb in 0..4 {
                            if sub_mb_types[smb] == 0 {
                                continue;
                            }
                            let (_, _, _, pl1) = B_SUB_TABLE[sub_mb_types[smb] as usize];
                            if pl1 && header.num_ref_idx_l1_active > 1 {
                                sub_ref_l1[smb] =
                                    reader.read_te(header.num_ref_idx_l1_active - 1)? as i8;
                            } else if pl1 {
                                sub_ref_l1[smb] = 0;
                            }
                        }

                        // Store ref_idx into cache for MV prediction (both lists)
                        for list in 0..2 {
                            let ref_store = if list == 0 {
                                &mut ref_idx_store_l0
                            } else {
                                &mut ref_idx_store_l1
                            };
                            let sub_ref = if list == 0 { &sub_ref_l0 } else { &sub_ref_l1 };
                            for smb in 0..4 {
                                let (sy, sx) = sub_mb_origins[smb];
                                for r in (0..8).step_by(4) {
                                    for c in (0..8).step_by(4) {
                                        let lr = (sy + r) / 4;
                                        let lc = (sx + c) / 4;
                                        if let Some(blk) = BLOCK_INDEX_TO_OFFSET
                                            .iter()
                                            .position(|&(br, bc)| br / 4 == lr && bc / 4 == lc)
                                        {
                                            ref_store[mb_idx * 16 + blk] = sub_ref[smb];
                                        }
                                    }
                                }
                            }
                        }

                        // Collect sub-partition layouts for each sub-MB
                        struct SubLayout {
                            smb: usize,
                            sx: usize,
                            sy: usize,
                            sub_w: usize,
                            sub_h: usize,
                            pl0: bool,
                            pl1: bool,
                            offsets: Vec<(usize, usize)>, // (dx, dy) within 8x8
                        }
                        let mut layouts: Vec<SubLayout> = Vec::new();
                        for smb in 0..4 {
                            let (sy, sx) = sub_mb_origins[smb];
                            let smt = sub_mb_types[smb] as usize;
                            if smt == 0 {
                                // B_Direct_8x8: handled separately below
                                layouts.push(SubLayout {
                                    smb,
                                    sx,
                                    sy,
                                    sub_w: 8,
                                    sub_h: 8,
                                    pl0: false,
                                    pl1: false, // direct mode flag
                                    offsets: vec![(0, 0)],
                                });
                                continue;
                            }
                            let (sub_w, sub_h, pl0, pl1) = B_SUB_TABLE[smt];
                            let offsets = match (sub_w, sub_h) {
                                (8, 8) => vec![(0, 0)],
                                (8, 4) => vec![(0, 0), (0, 4)],
                                (4, 8) => vec![(0, 0), (4, 0)],
                                (4, 4) => vec![(0, 0), (4, 0), (0, 4), (4, 4)],
                                _ => unreachable!(),
                            };
                            layouts.push(SubLayout {
                                smb,
                                sx,
                                sy,
                                sub_w,
                                sub_h,
                                pl0,
                                pl1,
                                offsets,
                            });
                        }

                        // Derive B_Direct_8x8 MVs per 4x4 block BEFORE MVD parsing
                        for layout in &layouts {
                            if sub_mb_types[layout.smb] == 0 {
                                make_ctx!().derive_direct_mvs(
                                    mb_idx,
                                    layout.smb * 4,
                                    4,
                                    header.direct_spatial_mv_pred_flag,
                                    sps.direct_8x8_inference_flag,
                                    current_poc,
                                    &_ref_pic_list_l0,
                                    &_ref_pic_list_l1,
                                );
                            }
                        }

                        // Parse MVDs: for each list, for each sub-MB, for each sub-partition
                        // (spec 7.3.5.1 parsing order)
                        struct SubMv {
                            mv_l0: [i16; 2],
                            mv_l1: [i16; 2],
                        }
                        let mut sub_mvs: Vec<SubMv> = Vec::new();
                        // Initialize with zeros
                        for layout in &layouts {
                            for &(_dx, _dy) in &layout.offsets {
                                sub_mvs.push(SubMv {
                                    mv_l0: [0; 2],
                                    mv_l1: [0; 2],
                                });
                            }
                        }

                        // L0 MVDs first
                        let mut idx = 0;
                        for layout in &layouts {
                            if sub_mb_types[layout.smb] == 0 {
                                idx += 1;
                                continue;
                            } // direct
                            for &(dx, dy) in &layout.offsets {
                                if layout.pl0 {
                                    let px = layout.sx + dx;
                                    let py = layout.sy + dy;
                                    let mvd_x = reader.read_se()? as i16;
                                    let mvd_y = reader.read_se()? as i16;
                                    let (mvp_x, mvp_y) = predict_mv_sub(
                                        &mv_store_l0,
                                        &ref_idx_store_l0,
                                        mb_idx,
                                        mb_width as usize,
                                        px,
                                        py,
                                        layout.sub_w,
                                        layout.sub_h,
                                        sub_ref_l0[layout.smb],
                                        &mb_slice_id,
                                        this_slice_id,
                                    );
                                    let mv = [mvp_x + mvd_x, mvp_y + mvd_y];
                                    sub_mvs[idx].mv_l0 = mv;
                                    // Store immediately for subsequent sub-partition prediction
                                    for r in (0..layout.sub_h).step_by(4) {
                                        for c in (0..layout.sub_w).step_by(4) {
                                            let lr = (py + r) / 4;
                                            let lc = (px + c) / 4;
                                            if let Some(blk) = BLOCK_INDEX_TO_OFFSET
                                                .iter()
                                                .position(|&(br, bc)| br / 4 == lr && bc / 4 == lc)
                                            {
                                                mv_store_l0[mb_idx * 16 + blk] = mv;
                                            }
                                        }
                                    }
                                }
                                idx += 1;
                            }
                        }

                        // L1 MVDs second
                        idx = 0;
                        for layout in &layouts {
                            if sub_mb_types[layout.smb] == 0 {
                                idx += 1;
                                continue;
                            }
                            for &(dx, dy) in &layout.offsets {
                                if layout.pl1 {
                                    let px = layout.sx + dx;
                                    let py = layout.sy + dy;
                                    let mvd_x = reader.read_se()? as i16;
                                    let mvd_y = reader.read_se()? as i16;
                                    let (mvp_x, mvp_y) = predict_mv_sub(
                                        &mv_store_l1,
                                        &ref_idx_store_l1,
                                        mb_idx,
                                        mb_width as usize,
                                        px,
                                        py,
                                        layout.sub_w,
                                        layout.sub_h,
                                        sub_ref_l1[layout.smb],
                                        &mb_slice_id,
                                        this_slice_id,
                                    );
                                    let mv = [mvp_x + mvd_x, mvp_y + mvd_y];
                                    sub_mvs[idx].mv_l1 = mv;
                                    for r in (0..layout.sub_h).step_by(4) {
                                        for c in (0..layout.sub_w).step_by(4) {
                                            let lr = (py + r) / 4;
                                            let lc = (px + c) / 4;
                                            if let Some(blk) = BLOCK_INDEX_TO_OFFSET
                                                .iter()
                                                .position(|&(br, bc)| br / 4 == lr && bc / 4 == lc)
                                            {
                                                mv_store_l1[mb_idx * 16 + blk] = mv;
                                            }
                                        }
                                    }
                                }
                                idx += 1;
                            }
                        }

                        // Build sub_parts from stored MVs
                        idx = 0;
                        for layout in &layouts {
                            if sub_mb_types[layout.smb] == 0 {
                                // B_Direct_8x8: coalesce to 8x8 if all 4 blocks share MV/ref
                                let base = mb_idx * 16;
                                let blk0 = BLOCK_INDEX_TO_OFFSET
                                    .iter()
                                    .position(|&(br, bc)| {
                                        br / 4 == layout.sy / 4 && bc / 4 == layout.sx / 4
                                    })
                                    .unwrap_or(0);
                                let mv0 = mv_store_l0[base + blk0];
                                let mv1 = mv_store_l1[base + blk0];
                                let r0 = ref_idx_store_l0[base + blk0];
                                let r1 = ref_idx_store_l1[base + blk0];
                                let uniform = (1..4).all(|sub| {
                                    let b = blk0 + sub;
                                    mv_store_l0[base + b] == mv0
                                        && mv_store_l1[base + b] == mv1
                                        && ref_idx_store_l0[base + b] == r0
                                        && ref_idx_store_l1[base + b] == r1
                                });
                                if uniform {
                                    sub_parts.push(SubPart {
                                        x: layout.sx,
                                        y: layout.sy,
                                        w: 8,
                                        h: 8,
                                        ref_idx_l0: r0,
                                        ref_idx_l1: r1,
                                        mv_l0: mv0,
                                        mv_l1: mv1,
                                        pred_l0: r0 >= 0,
                                        pred_l1: r1 >= 0,
                                    });
                                } else {
                                    for dr in (0..8).step_by(4) {
                                        for dc in (0..8).step_by(4) {
                                            let lr = (layout.sy + dr) / 4;
                                            let lc = (layout.sx + dc) / 4;
                                            let blk = BLOCK_INDEX_TO_OFFSET
                                                .iter()
                                                .position(|&(br, bc)| br / 4 == lr && bc / 4 == lc)
                                                .unwrap_or(0);
                                            let mv_l0 = mv_store_l0[base + blk];
                                            let mv_l1 = mv_store_l1[base + blk];
                                            let ri_l0 = ref_idx_store_l0[base + blk];
                                            let ri_l1 = ref_idx_store_l1[base + blk];
                                            sub_parts.push(SubPart {
                                                x: layout.sx + dc,
                                                y: layout.sy + dr,
                                                w: 4,
                                                h: 4,
                                                ref_idx_l0: ri_l0,
                                                ref_idx_l1: ri_l1,
                                                mv_l0,
                                                mv_l1,
                                                pred_l0: ri_l0 >= 0,
                                                pred_l1: ri_l1 >= 0,
                                            });
                                        }
                                    }
                                }
                                idx += 1;
                            } else {
                                let smt = sub_mb_types[layout.smb] as usize;
                                let (_, _, pl0, pl1) = B_SUB_TABLE[smt];
                                for &(dx, dy) in &layout.offsets {
                                    sub_parts.push(SubPart {
                                        x: layout.sx + dx,
                                        y: layout.sy + dy,
                                        w: layout.sub_w,
                                        h: layout.sub_h,
                                        ref_idx_l0: sub_ref_l0[layout.smb],
                                        ref_idx_l1: sub_ref_l1[layout.smb],
                                        mv_l0: sub_mvs[idx].mv_l0,
                                        mv_l1: sub_mvs[idx].mv_l1,
                                        pred_l0: pl0,
                                        pred_l1: pl1,
                                    });
                                    idx += 1;
                                }
                            }
                        }
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

                // 8x8 transform flag (High profile, spec 7.3.5)
                let use_8x8_dct_b = pps.transform_8x8_mode_flag
                    && cbp_luma != 0
                    && no_sub_less_8x8_b
                    && reader.read_bit()? != 0;

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
                if use_8x8_dct_b {
                    for i8x8 in 0..4 {
                        if cbp_luma & (1 << i8x8) == 0 {
                            for sub in 0..4 {
                                nc_luma[mb_idx * 16 + i8x8 * 4 + sub] = 0;
                            }
                            continue;
                        }
                        let mut block_coeffs = [0i32; 64];
                        let nc = compute_nc(
                            &nc_luma,
                            mb_idx,
                            mb_width as usize,
                            i8x8 * 4,
                            16,
                            &mb_slice_id,
                            this_slice_id,
                        );
                        let tc =
                            parse_residual_block_cavlc(&mut reader, &mut block_coeffs, 64, nc)?;
                        let tc_per = tc.div_ceil(4);
                        for sub in 0..4 {
                            nc_luma[mb_idx * 16 + i8x8 * 4 + sub] = tc_per;
                        }
                        let mut block_8x8 = [0i32; 64];
                        for i in 0..64 {
                            block_8x8[ZIGZAG_8X8_CAVLC[i]] = block_coeffs[i];
                        }
                        dequant_8x8(&mut block_8x8, qp_y, &pps.scaling_list_8x8[1]);
                        inverse_dct_8x8(&mut block_8x8);
                        let row_off = (i8x8 / 2) * 8;
                        let col_off = (i8x8 % 2) * 8;
                        for r in 0..8 {
                            for c in 0..8 {
                                luma_residual[(row_off + r) * 16 + col_off + c] =
                                    block_8x8[r * 8 + c];
                            }
                        }
                    }
                } else {
                    for blk in 0..16 {
                        if cbp_luma & (1 << (blk / 4)) != 0 {
                            let nc = compute_nc(
                                &nc_luma,
                                mb_idx,
                                mb_width as usize,
                                blk,
                                16,
                                &mb_slice_id,
                                this_slice_id,
                            );
                            let mut block_coeffs = [0i32; 16];
                            let tc =
                                parse_residual_block_cavlc(&mut reader, &mut block_coeffs, 16, nc)?;
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
                                    luma_residual[(blk_row + r) * 16 + blk_col + c] =
                                        raster[r * 4 + c];
                                }
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
                            ref_pic_safe(&_ref_pic_list_l0, sp.ref_idx_l0),
                            (mb_x + sp.x) as i32,
                            (mb_y + sp.y) as i32,
                            sp.mv_l0[0] as i32,
                            sp.mv_l0[1] as i32,
                            sp.w,
                            sp.h,
                            &mut pred_l0,
                        );
                        inter_pred::luma_mc(
                            ref_pic_safe(&_ref_pic_list_l1, sp.ref_idx_l1),
                            (mb_x + sp.x) as i32,
                            (mb_y + sp.y) as i32,
                            sp.mv_l1[0] as i32,
                            sp.mv_l1[1] as i32,
                            sp.w,
                            sp.h,
                            &mut pred_l1,
                        );
                        wctx.apply_bi(
                            &pred_l0,
                            &pred_l1,
                            &mut luma_pred,
                            sp.ref_idx_l0 as usize,
                            sp.ref_idx_l1 as usize,
                            false,
                            0,
                        );
                    } else if sp.pred_l0 {
                        inter_pred::luma_mc(
                            ref_pic_safe(&_ref_pic_list_l0, sp.ref_idx_l0),
                            (mb_x + sp.x) as i32,
                            (mb_y + sp.y) as i32,
                            sp.mv_l0[0] as i32,
                            sp.mv_l0[1] as i32,
                            sp.w,
                            sp.h,
                            &mut luma_pred,
                        );
                        if use_weight == 1 {
                            wctx.apply_uni(&mut luma_pred, 0, sp.ref_idx_l0 as usize, false, 0);
                        }
                    } else if sp.pred_l1 {
                        inter_pred::luma_mc(
                            ref_pic_safe(&_ref_pic_list_l1, sp.ref_idx_l1),
                            (mb_x + sp.x) as i32,
                            (mb_y + sp.y) as i32,
                            sp.mv_l1[0] as i32,
                            sp.mv_l1[1] as i32,
                            sp.w,
                            sp.h,
                            &mut luma_pred,
                        );
                        if use_weight == 1 {
                            wctx.apply_uni(&mut luma_pred, 1, sp.ref_idx_l1 as usize, false, 0);
                        }
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
                        let nc = compute_nc(
                            &nc_cb,
                            mb_idx,
                            mb_width as usize,
                            blk,
                            4,
                            &mb_slice_id,
                            this_slice_id,
                        );
                        let tc = parse_residual_block_cavlc(
                            &mut reader,
                            &mut chroma_ac_scan_cb[blk],
                            15,
                            nc,
                        )?;
                        nc_cb[mb_idx * 4 + blk] = tc;
                    }
                    for blk in 0..4 {
                        let nc = compute_nc(
                            &nc_cr,
                            mb_idx,
                            mb_width as usize,
                            blk,
                            4,
                            &mb_slice_id,
                            this_slice_id,
                        );
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
                    for sp in sub_parts.iter() {
                        let cx_off = sp.x / 2;
                        let cy_off = sp.y / 2;
                        let cw = sp.w.max(2) / 2;
                        let ch = sp.h.max(2) / 2;
                        if cw == 0 || ch == 0 {
                            continue;
                        }

                        let chroma_h = (height / 2) as usize;
                        let cx = (chroma_mb_x + cx_off) as i32;
                        let cy = (chroma_mb_y + cy_off) as i32;

                        let mut part_pred = vec![0u8; cw * ch];
                        if sp.pred_l0 && sp.pred_l1 {
                            let ref_l0 = ref_pic_safe(&_ref_pic_list_l0, sp.ref_idx_l0);
                            let ref_l1 = ref_pic_safe(&_ref_pic_list_l1, sp.ref_idx_l1);
                            let cr_l0 = if scale_idx == 4 { &ref_l0.u } else { &ref_l0.v };
                            let cr_l1 = if scale_idx == 4 { &ref_l1.u } else { &ref_l1.v };
                            let mut c_l0 = vec![0u8; cw * ch];
                            let mut c_l1 = vec![0u8; cw * ch];
                            inter_pred::chroma_mc(
                                cr_l0,
                                chroma_width,
                                chroma_h,
                                cx,
                                cy,
                                sp.mv_l0[0] as i32,
                                sp.mv_l0[1] as i32,
                                cw,
                                ch,
                                &mut c_l0,
                            );
                            inter_pred::chroma_mc(
                                cr_l1,
                                chroma_width,
                                chroma_h,
                                cx,
                                cy,
                                sp.mv_l1[0] as i32,
                                sp.mv_l1[1] as i32,
                                cw,
                                ch,
                                &mut c_l1,
                            );
                            let chroma_comp = if scale_idx == 4 { 0 } else { 1 };
                            wctx.apply_bi(
                                &c_l0,
                                &c_l1,
                                &mut part_pred,
                                sp.ref_idx_l0 as usize,
                                sp.ref_idx_l1 as usize,
                                true,
                                chroma_comp,
                            );
                        } else {
                            let (ref_list, ref_idx, mv) = if sp.pred_l0 {
                                (&_ref_pic_list_l0, sp.ref_idx_l0, sp.mv_l0)
                            } else {
                                (&_ref_pic_list_l1, sp.ref_idx_l1, sp.mv_l1)
                            };
                            let ref_pic = &ref_list[ref_idx as usize];
                            let cr = if scale_idx == 4 {
                                &ref_pic.u
                            } else {
                                &ref_pic.v
                            };
                            inter_pred::chroma_mc(
                                cr,
                                chroma_width,
                                chroma_h,
                                cx,
                                cy,
                                mv[0] as i32,
                                mv[1] as i32,
                                cw,
                                ch,
                                &mut part_pred,
                            );
                            if use_weight == 1 {
                                let chroma_comp = if scale_idx == 4 { 0 } else { 1 };
                                let (list, ri_val) = if sp.pred_l0 {
                                    (0, sp.ref_idx_l0 as usize)
                                } else {
                                    (1, sp.ref_idx_l1 as usize)
                                };
                                wctx.apply_uni(&mut part_pred, list, ri_val, true, chroma_comp);
                            }
                        }
                        for r in 0..ch {
                            for c in 0..cw {
                                chroma_pred[(cy_off + r) * 8 + cx_off + c] = part_pred[r * cw + c];
                            }
                        }
                    }

                    for y in 0..8 {
                        for x in 0..8 {
                            let val = (chroma_pred[y * 8 + x] as i32 + chroma_residual[y * 8 + x])
                                .clamp(0, 255) as u8;
                            frame_plane[(chroma_mb_y + y) * chroma_width + chroma_mb_x + x] = val;
                        }
                    }
                }

                mb_info[mb_idx] = MbInfo {
                    mb_type: MbType::Inter,
                    qp_y,
                    ..Default::default()
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
                    x: usize,
                    y: usize,
                    w: usize,
                    h: usize,
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
                                0 => vec![(0, 0, 8, 8)],                                           // 8x8
                                1 => vec![(0, 0, 8, 4), (0, 4, 8, 4)], // 8x4
                                2 => vec![(0, 0, 4, 8), (4, 0, 4, 8)], // 4x8
                                3 => vec![(0, 0, 4, 4), (4, 0, 4, 4), (0, 4, 4, 4), (4, 4, 4, 4)], // 4x4
                                _ => return Err(DecodeError::from("invalid sub_mb_type")),
                            };

                        for &(dx, dy, spw, sph) in &sub_parts_layout {
                            let px = sx + dx;
                            let py = sy + dy;
                            let mvd_x = reader.read_se()? as i16;
                            let mvd_y = reader.read_se()? as i16;
                            let (mvp_x, mvp_y) = predict_mv_sub(
                                &mv_store_l0,
                                &ref_idx_store_l0,
                                mb_idx,
                                mb_width as usize,
                                px,
                                py,
                                spw,
                                sph,
                                ref_idx,
                                &mb_slice_id,
                                this_slice_id,
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
                                x: px,
                                y: py,
                                w: spw,
                                h: sph,
                                ref_idx,
                                mv,
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
                            *ref_entry = reader.read_te(header.num_ref_idx_l0_active - 1)? as i8;
                        }
                    }

                    #[allow(clippy::needless_range_loop)]
                    for p in 0..num_parts {
                        let mvd_x = reader.read_se()? as i16;
                        let mvd_y = reader.read_se()? as i16;
                        let (mvp_x, mvp_y) = predict_mv(
                            &mv_store_l0,
                            &ref_idx_store_l0,
                            mb_idx,
                            mb_width as usize,
                            p,
                            part_w,
                            part_h,
                            part_ref[p],
                            &mb_slice_id,
                            this_slice_id,
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
                            x: px_off,
                            y: py_off,
                            w: part_w,
                            h: part_h,
                            ref_idx: part_ref[p],
                            mv,
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

                // 8x8 transform flag (High profile inter MBs, spec 7.3.5).
                // Only present when noSubMbPartSizeLessThan8x8Flag is true.
                let no_sub_less_8x8 = if is_p8x8 {
                    sub_mb_types.iter().all(|&smt| smt == 0) // P_8x8: smt=0 means 8x8 sub-partition
                } else {
                    true // P_16x16, P_16x8, P_8x16 always >= 8x8
                };
                let use_8x8_dct = pps.transform_8x8_mode_flag
                    && cbp_luma != 0
                    && no_sub_less_8x8
                    && reader.read_bit()? != 0;

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
                if use_8x8_dct {
                    // 8x8 transform: 4 blocks of 64 coefficients each
                    let scale_8x8 = if is_inter {
                        &pps.scaling_list_8x8[1]
                    } else {
                        &pps.scaling_list_8x8[0]
                    };
                    for i8x8 in 0..4 {
                        if cbp_luma & (1 << i8x8) == 0 {
                            continue;
                        }
                        let mut block_8x8 = [0i32; 64];
                        // Decode 4 groups of 16 coefficients via CAVLC
                        for i4x4 in 0..4 {
                            let blk = i8x8 * 4 + i4x4;
                            let nc = compute_nc(
                                &nc_luma,
                                mb_idx,
                                mb_width as usize,
                                blk,
                                16,
                                &mb_slice_id,
                                this_slice_id,
                            );
                            let mut quad_coeffs = [0i32; 16];
                            let tc =
                                parse_residual_block_cavlc(&mut reader, &mut quad_coeffs, 16, nc)?;
                            nc_luma[mb_idx * 16 + blk] = tc;
                            // Place into 8x8 block using scan table
                            let scan_base = i4x4 * 16;
                            for k in 0..16 {
                                if quad_coeffs[k] != 0 {
                                    block_8x8[ZIGZAG_8X8_CAVLC[scan_base + k]] = quad_coeffs[k];
                                }
                            }
                        }
                        dequant_8x8(&mut block_8x8, qp_y, scale_8x8);
                        inverse_dct_8x8(&mut block_8x8);
                        // Copy 8x8 residual into 16x16 luma residual
                        let row_off = (i8x8 / 2) * 8;
                        let col_off = (i8x8 % 2) * 8;
                        for r in 0..8 {
                            for c in 0..8 {
                                luma_residual[(row_off + r) * 16 + col_off + c] =
                                    block_8x8[r * 8 + c];
                            }
                        }
                    }
                } else {
                    // 4x4 transform (existing path)
                    for blk in 0..16 {
                        if cbp_luma & (1 << (blk / 4)) != 0 {
                            let nc = compute_nc(
                                &nc_luma,
                                mb_idx,
                                mb_width as usize,
                                blk,
                                16,
                                &mb_slice_id,
                                this_slice_id,
                            );
                            let mut block_coeffs = [0i32; 16];
                            let tc =
                                parse_residual_block_cavlc(&mut reader, &mut block_coeffs, 16, nc)?;
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
                                    luma_residual[(blk_row + r) * 16 + blk_col + c] =
                                        raster[r * 4 + c];
                                }
                            }
                        }
                    }
                } // close if use_8x8_dct else

                // Motion compensate and add residual for each sub-partition
                for sp in &sub_parts {
                    let ref_pic = &ref_pic_list[(sp.ref_idx as usize).min(ref_pic_list.len() - 1)];
                    let mut luma_pred = vec![0u8; sp.w * sp.h];
                    inter_pred::luma_mc(
                        ref_pic,
                        (mb_x + sp.x) as i32,
                        (mb_y + sp.y) as i32,
                        sp.mv[0] as i32,
                        sp.mv[1] as i32,
                        sp.w,
                        sp.h,
                        &mut luma_pred,
                    );
                    if use_weight == 1 {
                        wctx.apply_uni(&mut luma_pred, 0, sp.ref_idx as usize, false, 0);
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
                        let nc = compute_nc(
                            &nc_cb,
                            mb_idx,
                            mb_width as usize,
                            blk,
                            4,
                            &mb_slice_id,
                            this_slice_id,
                        );
                        let tc = parse_residual_block_cavlc(
                            &mut reader,
                            &mut chroma_ac_scan_cb[blk],
                            15,
                            nc,
                        )?;
                        nc_cb[mb_idx * 4 + blk] = tc;
                    }
                    for blk in 0..4 {
                        let nc = compute_nc(
                            &nc_cr,
                            mb_idx,
                            mb_width as usize,
                            blk,
                            4,
                            &mb_slice_id,
                            this_slice_id,
                        );
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
                            if cw == 0 || ch == 0 {
                                continue;
                            }
                            let part_ref_pic =
                                &ref_pic_list[(sp.ref_idx as usize).min(ref_pic_list.len() - 1)];
                            let chroma_ref = if scale_idx == 4 {
                                &part_ref_pic.u
                            } else {
                                &part_ref_pic.v
                            };
                            let mut part_pred = vec![0u8; cw * ch];
                            inter_pred::chroma_mc(
                                chroma_ref,
                                chroma_width,
                                (height / 2) as usize,
                                (chroma_mb_x + cx_off) as i32,
                                (chroma_mb_y + cy_off) as i32,
                                sp.mv[0] as i32,
                                sp.mv[1] as i32,
                                cw,
                                ch,
                                &mut part_pred,
                            );
                            if use_weight == 1 {
                                let chroma_comp = if scale_idx == 4 { 0 } else { 1 };
                                wctx.apply_uni(
                                    &mut part_pred,
                                    0,
                                    sp.ref_idx as usize,
                                    true,
                                    chroma_comp,
                                );
                            }
                            for r in 0..ch {
                                for c in 0..cw {
                                    chroma_pred[(cy_off + r) * 8 + cx_off + c] =
                                        part_pred[r * cw + c];
                                }
                            }
                        }

                        for y in 0..8 {
                            for x in 0..8 {
                                let val = (chroma_pred[y * 8 + x] as i32
                                    + chroma_residual[y * 8 + x])
                                    .clamp(0, 255) as u8;
                                frame_plane[(chroma_mb_y + y) * chroma_width + chroma_mb_x + x] =
                                    val;
                            }
                        }
                    }
                }

                mb_info[mb_idx] = MbInfo {
                    mb_type: MbType::Inter,
                    qp_y,
                    ..Default::default()
                };
                mb_idx += 1;
                continue;
            }

            // === Intra macroblock (I4x4, I16x16, I_PCM) ===
            // Cross-slice intra prediction: neighbors from other slices unavailable (spec 6.4.1)
            let above_mb_avail = mb_idx >= mb_width as usize
                && mb_slice_id[mb_idx - mb_width as usize] == this_slice_id;
            let left_mb_avail = !mb_idx.is_multiple_of(mb_width as usize)
                && mb_slice_id[mb_idx - 1] == this_slice_id;
            let above_left_mb_avail = mb_idx >= mb_width as usize
                && !mb_idx.is_multiple_of(mb_width as usize)
                && mb_slice_id[mb_idx - mb_width as usize - 1] == this_slice_id;
            let above_right_mb_avail = mb_idx >= mb_width as usize
                && (mb_idx % mb_width as usize) + 1 < mb_width as usize
                && mb_slice_id[mb_idx - mb_width as usize + 1] == this_slice_id;

            // Variables shared between I4x4/I16x16 for chroma reconstruction
            let intra_chroma_pred_mode;
            let cbp_chroma: u8;
            let qp_y;
            let qp_c;

            if mb_type == 0 {
                // === I_NxN (I4x4 or I8x8) macroblock ===

                // Check 8x8 transform flag
                let use_8x8_intra = pps.transform_8x8_mode_flag && reader.read_bit()? != 0;

                // Parse prediction modes: 4 for I8x8, 16 for I4x4
                let num_modes = if use_8x8_intra { 4 } else { 16 };
                let mut pred_modes = [2u8; 16];
                for blk_idx in 0..num_modes {
                    let blk = if use_8x8_intra { blk_idx * 4 } else { blk_idx };
                    let prev_flag = reader.read_bit()?;
                    let predicted = predict_i4x4_mode(
                        &i4x4_modes,
                        mb_idx,
                        mb_width as usize,
                        blk,
                        &mb_slice_id,
                        this_slice_id,
                    );
                    let mode = if prev_flag != 0 {
                        predicted
                    } else {
                        let rem = reader.read_bits(3)? as u8;
                        if rem < predicted {
                            rem
                        } else {
                            rem + 1
                        }
                    };
                    if use_8x8_intra {
                        // Store same mode for all 4 sub-blocks
                        for sub in 0..4 {
                            pred_modes[blk_idx * 4 + sub] = mode;
                            i4x4_modes[mb_idx * 16 + blk_idx * 4 + sub] = mode;
                        }
                    } else {
                        pred_modes[blk] = mode;
                        i4x4_modes[mb_idx * 16 + blk] = mode;
                    }
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

                // Parse luma residual and reconstruct
                if use_8x8_intra {
                    // I8x8: decode 4 8x8 blocks with 8x8 transform
                    let mut luma_residual = [0i32; 256];
                    for i8x8 in 0..4 {
                        if cbp_luma & (1 << i8x8) == 0 {
                            continue;
                        }
                        let mut block_8x8 = [0i32; 64];
                        for i4x4 in 0..4 {
                            let blk = i8x8 * 4 + i4x4;
                            let nc = compute_nc(
                                &nc_luma,
                                mb_idx,
                                mb_width as usize,
                                blk,
                                16,
                                &mb_slice_id,
                                this_slice_id,
                            );
                            let mut quad_coeffs = [0i32; 16];
                            let tc =
                                parse_residual_block_cavlc(&mut reader, &mut quad_coeffs, 16, nc)?;
                            nc_luma[mb_idx * 16 + blk] = tc;
                            let scan_base = i4x4 * 16;
                            for k in 0..16 {
                                if quad_coeffs[k] != 0 {
                                    block_8x8[ZIGZAG_8X8_CAVLC[scan_base + k]] = quad_coeffs[k];
                                }
                            }
                        }
                        dequant_8x8(&mut block_8x8, qp_y, &pps.scaling_list_8x8[0]);
                        inverse_dct_8x8(&mut block_8x8);

                        let row_off = (i8x8 / 2) * 8;
                        let col_off = (i8x8 % 2) * 8;
                        for r in 0..8 {
                            for c in 0..8 {
                                luma_residual[(row_off + r) * 16 + col_off + c] =
                                    block_8x8[r * 8 + c];
                            }
                        }
                    }
                    // I8x8 prediction + residual reconstruction
                    for i8x8 in 0..4 {
                        let row_off = (i8x8 / 2) * 8;
                        let col_off = (i8x8 % 2) * 8;
                        let px = mb_x + col_off;
                        let py = mb_y + row_off;

                        // Gather reference samples: 16 above (8 + 8 above-right)
                        let above_avail_8 = py > 0 && (row_off > 0 || above_mb_avail);
                        let above_buf: Option<[u8; 16]> = if above_avail_8 {
                            let mut buf = [0u8; 16];
                            for (i, b) in buf.iter_mut().enumerate().take(8) {
                                *b = frame.y[(py - 1) * stride + px + i];
                            }
                            // Above-right: available if at top of MB or from MB above
                            let has_tr = if row_off == 0 {
                                if px + 8 < (mb_x + 16).min(stride) {
                                    true
                                } else if px + 8 < stride {
                                    above_right_mb_avail
                                } else {
                                    false
                                }
                            } else {
                                col_off == 0 // only top-left 8x8 has above-right within MB
                            };
                            if has_tr {
                                for i in 0..8 {
                                    let col = (px + 8 + i).min(stride - 1);
                                    buf[8 + i] = frame.y[(py - 1) * stride + col];
                                }
                            } else {
                                let last = buf[7];
                                buf[8..].fill(last);
                            }
                            Some(buf)
                        } else {
                            None
                        };

                        let left_avail_8 = px > 0 && (col_off > 0 || left_mb_avail);
                        let left_buf: Option<[u8; 8]> = if left_avail_8 {
                            let mut buf = [0u8; 8];
                            for (i, b) in buf.iter_mut().enumerate() {
                                *b = frame.y[(py + i) * stride + px - 1];
                            }
                            Some(buf)
                        } else {
                            None
                        };

                        let al_avail_8 = px > 0
                            && py > 0
                            && ((row_off > 0 && col_off > 0)
                                || (row_off > 0 && col_off == 0 && left_mb_avail)
                                || (row_off == 0 && col_off > 0 && above_mb_avail)
                                || (row_off == 0 && col_off == 0 && above_left_mb_avail));
                        let above_left_val = if al_avail_8 {
                            Some(frame.y[(py - 1) * stride + px - 1])
                        } else {
                            None
                        };

                        let has_topright = above_buf.is_some()
                            && ((row_off == 0 && px + 8 < stride) || (row_off > 0 && col_off == 0));

                        let mode = pred_modes[i8x8 * 4]; // mode stored per 8x8
                        let mut pred = [0u8; 64];
                        predict_intra_8x8(
                            mode,
                            above_buf.as_ref().map(|b| &b[..]),
                            left_buf.as_ref().map(|b| &b[..]),
                            above_left_val,
                            has_topright,
                            &mut pred,
                        );

                        // Add residual and write to frame
                        for r in 0..8 {
                            for c in 0..8 {
                                let val = (pred[r * 8 + c] as i32
                                    + luma_residual[(row_off + r) * 16 + col_off + c])
                                    .clamp(0, 255) as u8;
                                frame.y[(py + r) * stride + px + c] = val;
                            }
                        }
                    }
                } else {
                    // I4x4: existing 4x4 path
                    for blk in 0..16 {
                        let (blk_row, blk_col) = BLOCK_INDEX_TO_OFFSET[blk];
                        let px = mb_x + blk_col;
                        let py = mb_y + blk_row;

                        // Parse residual
                        let mut block_coeffs = [0i32; 16];
                        if cbp_luma & (1 << (blk / 4)) != 0 {
                            let nc = compute_nc(
                                &nc_luma,
                                mb_idx,
                                mb_width as usize,
                                blk,
                                16,
                                &mb_slice_id,
                                this_slice_id,
                            );
                            let tc =
                                parse_residual_block_cavlc(&mut reader, &mut block_coeffs, 16, nc)?;
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
                        let local_row = py - mb_y;
                        let local_col = px - mb_x;
                        let above_avail = py > 0 && (local_row > 0 || above_mb_avail);
                        let above_buf: Option<[u8; 8]> = if above_avail {
                            let mut buf = [0u8; 8];
                            for (i, b) in buf.iter_mut().enumerate().take(4) {
                                *b = frame.y[(py - 1) * stride + px + i];
                            }
                            let topright_avail = if local_row == 0 {
                                if px + 4 < (mb_x + 16).min(stride) {
                                    true // within same MB's above neighbor
                                } else if px + 4 < stride {
                                    above_right_mb_avail
                                } else {
                                    false
                                }
                            } else {
                                !matches!(blk, 3 | 7 | 11 | 13 | 15)
                            };
                            if topright_avail {
                                for (i, b) in buf.iter_mut().enumerate().skip(4) {
                                    let col = (px + i).min(stride - 1);
                                    *b = frame.y[(py - 1) * stride + col];
                                }
                            } else {
                                let last = buf[3];
                                buf[4..8].fill(last);
                            }
                            Some(buf)
                        } else {
                            None
                        };
                        let left_avail = px > 0 && (local_col > 0 || left_mb_avail);
                        let left_buf: Option<[u8; 4]> = if left_avail {
                            let mut buf = [0u8; 4];
                            for (i, b) in buf.iter_mut().enumerate() {
                                *b = frame.y[(py + i) * stride + px - 1];
                            }
                            Some(buf)
                        } else {
                            None
                        };
                        let al_avail = px > 0
                            && py > 0
                            && ((local_row > 0 && local_col > 0)
                                || (local_row > 0 && local_col == 0 && left_mb_avail)
                                || (local_row == 0 && local_col > 0 && above_mb_avail)
                                || (local_row == 0 && local_col == 0 && above_left_mb_avail));
                        let above_left_val = if al_avail {
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
                } // close if use_8x8_intra else
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
                let nc_dc = compute_nc(
                    &nc_luma,
                    mb_idx,
                    mb_width as usize,
                    0,
                    16,
                    &mb_slice_id,
                    this_slice_id,
                );
                parse_residual_block_cavlc(&mut reader, &mut luma_dc, 16, nc_dc)?;

                // Parse luma AC
                let mut luma_ac_scan = [[0i32; 15]; 16];
                if cbp_luma != 0 {
                    for blk in 0..16 {
                        let nc = compute_nc(
                            &nc_luma,
                            mb_idx,
                            mb_width as usize,
                            blk,
                            16,
                            &mb_slice_id,
                            this_slice_id,
                        );
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

                const DC_RASTER_TO_BLOCK: [usize; 16] =
                    [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15];

                let mut luma_residual = [0i32; 256];
                for blk in 0..16 {
                    let (blk_row, blk_col) = BLOCK_INDEX_TO_OFFSET[blk];
                    let mut block_raster = [0i32; 16];
                    let dc_idx = DC_RASTER_TO_BLOCK.iter().position(|&b| b == blk).unwrap();
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

                // I16x16 prediction + residual
                make_ctx!().reconstruct_luma_16x16(
                    mb_x,
                    mb_y,
                    intra16x16_pred_mode,
                    &luma_residual,
                    above_mb_avail,
                    left_mb_avail,
                    above_left_mb_avail,
                );
            } else if mb_type == 25 {
                // === I_PCM macroblock ===
                reader.align_to_byte();
                for r in 0..16 {
                    for c in 0..16 {
                        frame.y[(mb_y + r) * stride + mb_x + c] = reader.read_bits(8)? as u8;
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
                    ..Default::default()
                };
                prev_mb_qp = 0;
                continue;
            } else {
                return Err(DecodeError::from("unsupported mb_type for I slice"));
            }

            mb_info[mb_idx] = MbInfo {
                mb_type: MbType::Intra,
                qp_y,
                ..Default::default()
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
                    let nc = compute_nc(
                        &nc_cb,
                        mb_idx,
                        mb_width as usize,
                        blk,
                        4,
                        &mb_slice_id,
                        this_slice_id,
                    );
                    let tc = parse_residual_block_cavlc(
                        &mut reader,
                        &mut chroma_ac_scan_cb[blk],
                        15,
                        nc,
                    )?;
                    nc_cb[mb_idx * 4 + blk] = tc;
                }
                for blk in 0..4 {
                    let nc = compute_nc(
                        &nc_cr,
                        mb_idx,
                        mb_width as usize,
                        blk,
                        4,
                        &mb_slice_id,
                        this_slice_id,
                    );
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
                let above_c: Option<Vec<u8>> = if chroma_mb_y > 0 && above_mb_avail {
                    Some(
                        (0..8)
                            .map(|x| plane_buf[(chroma_mb_y - 1) * chroma_width + chroma_mb_x + x])
                            .collect(),
                    )
                } else {
                    None
                };
                let left_c: Option<Vec<u8>> = if chroma_mb_x > 0 && left_mb_avail {
                    Some(
                        (0..8)
                            .map(|y| plane_buf[(chroma_mb_y + y) * chroma_width + chroma_mb_x - 1])
                            .collect(),
                    )
                } else {
                    None
                };
                let above_left_c = if chroma_mb_x > 0 && chroma_mb_y > 0 && above_left_mb_avail {
                    Some(plane_buf[(chroma_mb_y - 1) * chroma_width + chroma_mb_x - 1])
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
                        let val = (chroma_pred[y * 8 + x] as i32 + chroma_residual[y * 8 + x])
                            .clamp(0, 255) as u8;
                        plane_buf[(chroma_mb_y + y) * chroma_width + chroma_mb_x + x] = val;
                    }
                }
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
            is_p_slice,
            is_b_slice,
            &ref_pic_list,
            &_ref_pic_list_l0,
            &_ref_pic_list_l1,
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
