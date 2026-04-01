//! Per-slice mutable decode state shared across all MB decode paths.
//!
//! `SliceContext` bundles the ~25 mutable arrays and geometry values that
//! every MB decoder (CABAC/CAVLC, I/P/B) reads and writes. Extracting it
//! from `decode_slice` enables splitting MB decode logic into methods.

use std::rc::Rc;

use crate::decoder::Frame;
use crate::dpb::DecodedPicture;
use crate::inter_pred;
use crate::mv_pred::{predict_mv_skip, WeightContext};

/// Mutable per-slice state passed to MB decode routines.
///
/// All mutable MB-level arrays live here so that individual decode
/// functions can be extracted as methods on `&mut SliceContext`.
#[allow(dead_code)] // Fields progressively used as methods are extracted from decode_slice
pub(crate) struct SliceContext<'a> {
    // Pixel output
    pub frame: &'a mut Frame,
    pub stride: usize,
    pub width: u32,
    pub height: u32,

    // Geometry
    pub mb_width: u32,

    // Per-4x4-block coefficient counts (for CABAC CBF / CAVLC nC)
    pub nc_luma: &'a mut [u8],
    pub nc_cb: &'a mut [u8],
    pub nc_cr: &'a mut [u8],

    // Per-4x4-block motion vectors and reference indices
    pub mv_store_l0: &'a mut [[i16; 2]],
    pub mv_store_l1: &'a mut [[i16; 2]],
    pub ref_idx_store_l0: &'a mut [i8],
    pub ref_poc_store_l0: &'a mut [i32],
    pub ref_idx_store_l1: &'a mut [i8],

    // Per-4x4-block MVD (for CABAC amvd context)
    pub mvd_store: &'a mut [[i16; 2]],
    pub mvd_store_l1: &'a mut [[i16; 2]],

    // Per-MB deblocking / neighbor info
    pub mb_info: &'a mut [crate::deblock::MbInfo],

    // Per-4x4-block intra prediction modes
    pub i4x4_modes: &'a mut [u8],

    // Per-MB CABAC neighbor context state
    pub mb_cbp: &'a mut [u16],
    pub mb_chroma_pred: &'a mut [u8],
    pub mb_is_8x8dct: &'a mut [bool],
    pub mb_skip: &'a mut [bool],
    pub mb_is_direct: &'a mut [bool],
    pub is_i16x16: &'a mut [bool],

    // Multi-slice boundary tracking
    pub mb_slice_id: &'a mut [u16],
    pub this_slice_id: u16,

    // QP state (carried across MBs within a slice)
    pub prev_mb_qp: i32,
    pub last_qp_delta_nonzero: bool,
}

impl SliceContext<'_> {
    /// Decode a P-slice skip macroblock: median MV prediction, MC, no residual.
    pub(crate) fn decode_p_skip_mb(
        &mut self,
        mb_idx: usize,
        mb_x: usize,
        mb_y: usize,
        ref_pic_list: &[Rc<DecodedPicture>],
        wctx: &WeightContext,
        use_weight: u8,
    ) {
        let (mvp_x, mvp_y) = predict_mv_skip(
            self.mv_store_l0,
            self.ref_idx_store_l0,
            mb_idx,
            self.mb_width as usize,
            self.mb_slice_id,
            self.this_slice_id,
        );
        if let Some(ref_pic) = ref_pic_list.first() {
            // Luma MC
            let mut luma_pred = [0u8; 256];
            inter_pred::luma_mc(
                ref_pic,
                mb_x as i32,
                mb_y as i32,
                mvp_x as i32,
                mvp_y as i32,
                16,
                16,
                &mut luma_pred,
            );
            if use_weight == 1 {
                wctx.apply_uni(&mut luma_pred, 0, 0, false, 0);
            }
            for r in 0..16 {
                for c in 0..16 {
                    self.frame.y[(mb_y + r) * self.stride + mb_x + c] = luma_pred[r * 16 + c];
                }
            }
            // Chroma MC
            let cw = (self.width / 2) as usize;
            let cx = mb_x / 2;
            let cy = mb_y / 2;
            let mut cb_pred = [0u8; 64];
            let mut cr_pred = [0u8; 64];
            inter_pred::chroma_mc(
                &ref_pic.u,
                cw,
                (self.height / 2) as usize,
                cx as i32,
                cy as i32,
                mvp_x as i32,
                mvp_y as i32,
                8,
                8,
                &mut cb_pred,
            );
            inter_pred::chroma_mc(
                &ref_pic.v,
                cw,
                (self.height / 2) as usize,
                cx as i32,
                cy as i32,
                mvp_x as i32,
                mvp_y as i32,
                8,
                8,
                &mut cr_pred,
            );
            if use_weight == 1 {
                wctx.apply_uni(&mut cb_pred, 0, 0, true, 0);
                wctx.apply_uni(&mut cr_pred, 0, 0, true, 1);
            }
            for r in 0..8 {
                for c in 0..8 {
                    self.frame.u[(cy + r) * cw + cx + c] = cb_pred[r * 8 + c];
                    self.frame.v[(cy + r) * cw + cx + c] = cr_pred[r * 8 + c];
                }
            }
        }
        // Store MVs and ref indices
        for blk in 0..16 {
            self.mv_store_l0[mb_idx * 16 + blk] = [mvp_x, mvp_y];
            self.ref_idx_store_l0[mb_idx * 16 + blk] = 0;
        }
    }

    /// Fill per-4x4-block MV/ref/nnz data into MbInfo for deblocking bS derivation,
    /// and build the per-block ref POC table for temporal direct mode.
    ///
    /// Called once after the MB loop completes, covering MBs `first_mb..last_mb`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn finalize_mb_info(
        &mut self,
        first_mb: usize,
        last_mb: usize,
        is_p_slice: bool,
        is_b_slice: bool,
        ref_pic_list: &[Rc<DecodedPicture>],
        ref_pic_list_l0: &[Rc<DecodedPicture>],
        ref_pic_list_l1: &[Rc<DecodedPicture>],
    ) {
        let list_count = if is_b_slice {
            2u8
        } else if is_p_slice {
            1
        } else {
            0
        };
        let l0_list = if is_p_slice {
            ref_pic_list
        } else {
            ref_pic_list_l0
        };

        #[allow(clippy::needless_range_loop)]
        for mi in first_mb..last_mb {
            let info = &mut self.mb_info[mi];
            let base = mi * 16;
            info.list_count = list_count;
            for blk in 0..16 {
                info.mv_l0[blk] = self.mv_store_l0[base + blk];
                info.ref_idx_l0[blk] = self.ref_idx_store_l0[base + blk];
                info.mv_l1[blk] = self.mv_store_l1[base + blk];
                info.ref_idx_l1[blk] = self.ref_idx_store_l1[base + blk];
                info.nnz[blk] = self.nc_luma[base + blk] > 0;
                let ri_l0 = self.ref_idx_store_l0[base + blk];
                info.ref_poc_l0[blk] = if ri_l0 >= 0 {
                    l0_list
                        .get(ri_l0 as usize)
                        .map(|p| p.pic_order_cnt)
                        .unwrap_or(-1)
                } else {
                    -1
                };
                let ri_l1 = self.ref_idx_store_l1[base + blk];
                info.ref_poc_l1[blk] = if ri_l1 >= 0 {
                    ref_pic_list_l1
                        .get(ri_l1 as usize)
                        .map(|p| p.pic_order_cnt)
                        .unwrap_or(-1)
                } else {
                    -1
                };
            }
        }

        // Build per-block ref POC table for temporal direct mode (spec 8.4.1.2.3).
        for i in 0..self.ref_poc_store_l0.len() {
            let ri = self.ref_idx_store_l0[i];
            if ri >= 0 && (ri as usize) < l0_list.len() {
                self.ref_poc_store_l0[i] = l0_list[ri as usize].pic_order_cnt;
            }
        }
    }
}
