//! Motion vector prediction and direct mode derivation.
//!
//! Contains MV prediction (median, directional), spatial/temporal direct mode,
//! MV neighbor lookups, skip MV derivation, and weighted prediction context.

use std::rc::Rc;

use crate::dpb::DecodedPicture;
use crate::inter_pred;
use crate::residual::OFFSET_TO_BLOCK;
use crate::slice::PredWeightTable;

#[allow(clippy::too_many_arguments)]
pub(crate) fn predict_mv_sub(
    mv_store_l0: &[[i16; 2]],
    ref_idx_store_l0: &[i8],
    mb_idx: usize,
    mb_width: usize,
    px: usize,
    py: usize,
    spw: usize,
    _sph: usize,
    ref_idx: i8,
    mb_slice_id: &[u16],
    cur_slice_id: u16,
) -> (i16, i16) {
    // Reuse the general predict_mv with the sub-partition's position and size.
    // The neighbor lookup functions already handle arbitrary py_off/px_off.
    let a = get_mv_neighbor_left(
        mv_store_l0,
        ref_idx_store_l0,
        mb_idx,
        mb_width,
        py,
        px,
        mb_slice_id,
        cur_slice_id,
    );
    let b = get_mv_neighbor_above(
        mv_store_l0,
        ref_idx_store_l0,
        mb_idx,
        mb_width,
        py,
        px,
        mb_slice_id,
        cur_slice_id,
    );
    let c = get_mv_neighbor_above_right(
        mv_store_l0,
        ref_idx_store_l0,
        mb_idx,
        mb_width,
        py,
        px,
        spw,
        mb_slice_id,
        cur_slice_id,
    )
    .or_else(|| {
        get_mv_neighbor_above_left(
            mv_store_l0,
            ref_idx_store_l0,
            mb_idx,
            mb_width,
            py,
            px,
            mb_slice_id,
            cur_slice_id,
        )
    });

    // match_count directional logic (same as predict_mv)
    let ref_a = a.map(|(_, r)| r).unwrap_or(-1);
    let ref_b = b.map(|(_, r)| r).unwrap_or(-1);
    let ref_c = c.map(|(_, r)| r).unwrap_or(-1);
    let match_count =
        (ref_a == ref_idx) as u8 + (ref_b == ref_idx) as u8 + (ref_c == ref_idx) as u8;

    if match_count == 1 {
        if ref_a == ref_idx {
            if let Some((mv, _)) = a {
                return (mv[0], mv[1]);
            }
        }
        if ref_b == ref_idx {
            if let Some((mv, _)) = b {
                return (mv[0], mv[1]);
            }
        }
        if ref_c == ref_idx {
            if let Some((mv, _)) = c {
                return (mv[0], mv[1]);
            }
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

/// Safely index a ref pic list, clamping out-of-range indices to the last entry.
#[inline]
pub(crate) fn ref_pic_safe(list: &[Rc<DecodedPicture>], idx: i8) -> &Rc<DecodedPicture> {
    &list[(idx as usize).min(list.len() - 1)]
}

/// Bundles weighted prediction parameters for a slice, avoiding long argument lists.
pub(crate) struct WeightContext<'a> {
    pub(crate) use_weight: u8,
    pub(crate) wt: Option<&'a PredWeightTable>,
    pub(crate) implicit_weights: &'a [Vec<i32>],
}

impl WeightContext<'_> {
    /// Apply weighted uni-prediction in place. No-op if use_weight != 1.
    pub(crate) fn apply_uni(
        &self,
        pred: &mut [u8],
        list: usize,
        ref_idx: usize,
        is_chroma: bool,
        chroma_comp: usize,
    ) {
        if self.use_weight != 1 {
            return;
        }
        let wt = match self.wt {
            Some(w) => w,
            None => return,
        };
        let refs = if list == 0 { &wt.l0 } else { &wt.l1 };
        if ref_idx >= refs.len() {
            return;
        }
        let rw = &refs[ref_idx];
        if is_chroma {
            inter_pred::weighted_uni(
                pred,
                wt.chroma_log2_weight_denom,
                rw.chroma_weight[chroma_comp],
                rw.chroma_offset[chroma_comp],
            );
        } else {
            inter_pred::weighted_uni(
                pred,
                wt.luma_log2_weight_denom,
                rw.luma_weight,
                rw.luma_offset,
            );
        }
    }

    /// Apply weighted bi-prediction, replacing the standard (a+b+1)>>1 average.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apply_bi(
        &self,
        pred_l0: &[u8],
        pred_l1: &[u8],
        output: &mut [u8],
        ref_idx_l0: usize,
        ref_idx_l1: usize,
        is_chroma: bool,
        chroma_comp: usize,
    ) {
        match self.use_weight {
            1 => {
                let wt = match self.wt {
                    Some(w) => w,
                    None => {
                        inter_pred::bi_pred_avg(pred_l0, pred_l1, output);
                        return;
                    }
                };
                let rw0 = wt.l0.get(ref_idx_l0);
                let rw1 = wt.l1.get(ref_idx_l1);
                match (rw0, rw1) {
                    (Some(w0), Some(w1)) if is_chroma => {
                        inter_pred::weighted_bi(
                            pred_l0,
                            pred_l1,
                            output,
                            wt.chroma_log2_weight_denom,
                            w0.chroma_weight[chroma_comp],
                            w0.chroma_offset[chroma_comp],
                            w1.chroma_weight[chroma_comp],
                            w1.chroma_offset[chroma_comp],
                        );
                    }
                    (Some(w0), Some(w1)) => {
                        inter_pred::weighted_bi(
                            pred_l0,
                            pred_l1,
                            output,
                            wt.luma_log2_weight_denom,
                            w0.luma_weight,
                            w0.luma_offset,
                            w1.luma_weight,
                            w1.luma_offset,
                        );
                    }
                    _ => inter_pred::bi_pred_avg(pred_l0, pred_l1, output),
                }
            }
            2 => {
                if ref_idx_l0 < self.implicit_weights.len()
                    && ref_idx_l1 < self.implicit_weights[ref_idx_l0].len()
                {
                    let w0 = self.implicit_weights[ref_idx_l0][ref_idx_l1];
                    inter_pred::weighted_bi_implicit(pred_l0, pred_l1, output, w0, 64 - w0);
                } else {
                    inter_pred::bi_pred_avg(pred_l0, pred_l1, output);
                }
            }
            _ => inter_pred::bi_pred_avg(pred_l0, pred_l1, output),
        }
    }
}

/// Motion vector prediction for P_Skip macroblocks (spec 8.4.1.1).
///
/// Returns (0, 0) if either neighbor A (left) or B (above) is unavailable
/// or has ref_idx=0 with zero MV. Otherwise uses the standard median predictor.
pub(crate) fn predict_mv_skip(
    mv_store_l0: &[[i16; 2]],
    ref_idx_store_l0: &[i8],
    mb_idx: usize,
    mb_width: usize,
    mb_slice_id: &[u16],
    cur_slice_id: u16,
) -> (i16, i16) {
    let a = get_mv_neighbor_left(
        mv_store_l0,
        ref_idx_store_l0,
        mb_idx,
        mb_width,
        0,
        0,
        mb_slice_id,
        cur_slice_id,
    );
    let b = get_mv_neighbor_above(
        mv_store_l0,
        ref_idx_store_l0,
        mb_idx,
        mb_width,
        0,
        0,
        mb_slice_id,
        cur_slice_id,
    );

    // Spec 8.4.1.1: if A is unavailable or (refA==0 && mvA==(0,0)), OR
    // if B is unavailable or (refB==0 && mvB==(0,0)), then skip MV = (0,0).
    let a_zero = match a {
        None => true,
        Some((mv, ri)) => ri == 0 && mv[0] == 0 && mv[1] == 0,
    };
    let b_zero = match b {
        None => true,
        Some((mv, ri)) => ri == 0 && mv[0] == 0 && mv[1] == 0,
    };
    if a_zero || b_zero {
        return (0, 0);
    }

    predict_mv(
        mv_store_l0,
        ref_idx_store_l0,
        mb_idx,
        mb_width,
        0,
        16,
        16,
        0,
        mb_slice_id,
        cur_slice_id,
    )
}

/// Derive spatial direct mode MVs with per-4x4-block co-located check.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub(crate) fn derive_spatial_direct_blk(
    mv_store_l0: &[[i16; 2]],
    ref_idx_store_l0: &[i8],
    mv_store_l1: &[[i16; 2]],
    ref_idx_store_l1: &[i8],
    mb_idx: usize,
    mb_width: usize,
    col_pic: Option<&DecodedPicture>,
    col_blk: usize,
    mb_slice_id: &[u16],
    cur_slice_id: u16,
    direct_8x8_inference_flag: bool,
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

        let a = get_mv_neighbor_left(
            mv_s,
            ref_s,
            mb_idx,
            mb_width,
            0,
            0,
            mb_slice_id,
            cur_slice_id,
        );
        let b = get_mv_neighbor_above(
            mv_s,
            ref_s,
            mb_idx,
            mb_width,
            0,
            0,
            mb_slice_id,
            cur_slice_id,
        );
        let c = get_mv_neighbor_above_right(
            mv_s,
            ref_s,
            mb_idx,
            mb_width,
            0,
            0,
            16,
            mb_slice_id,
            cur_slice_id,
        )
        .or_else(|| {
            get_mv_neighbor_above_left(
                mv_s,
                ref_s,
                mb_idx,
                mb_width,
                0,
                0,
                mb_slice_id,
                cur_slice_id,
            )
        });

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
            let match_count =
                (ref_a == min_ref) as u8 + (ref_b == min_ref) as u8 + (ref_c == min_ref) as u8;

            if match_count == 1 {
                // Use the single matching neighbor's MV
                if ref_a == min_ref {
                    if let Some((m, _)) = a {
                        mv[list] = m;
                        continue;
                    }
                }
                if ref_b == min_ref {
                    if let Some((m, _)) = b {
                        mv[list] = m;
                        continue;
                    }
                }
                if ref_c == min_ref {
                    if let Some((m, _)) = c {
                        mv[list] = m;
                        continue;
                    }
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

    // Co-located zero-MV refinement (spec 8.4.1.2.2):
    // Derive mvCol/refIdxCol from the co-located partition in ColPic (= RefPicList1[0]):
    //   - If co-located PredFlagL0 = 1: use L0 MV/ref
    //   - If co-located PredFlagL0 = 0 (L1-only, e.g. B_L1_16x16): use L1 MV/ref
    // Then colZeroFlag = 1 if refIdxCol maps to the same picture as
    // RefPicList0[0] and |mvCol| <= 1 in both components.
    // When colZeroFlag is set, zero out spatial MVs for lists where ref_idx == 0.
    // When direct_8x8_inference_flag is set, use the representative 4x4 block
    // per 8x8 group (same mapping as temporal direct).
    if let Some(col) = col_pic {
        let effective_blk = if direct_8x8_inference_flag {
            const INFERENCE_MAP: [usize; 4] = [0, 5, 10, 15];
            INFERENCE_MAP[col_blk / 4]
        } else {
            col_blk
        };
        let col_pos = mb_idx * 16 + effective_blk;
        if col_pos < col.ref_idx_l0.len() && !col.is_intra {
            let col_ref_l0 = col.ref_idx_l0[col_pos];
            // Determine which co-located MV to use for the zero check:
            // If co-located has L0 prediction (ref_idx_l0 >= 0): use L0 MV/ref
            // If co-located is L1-only (ref_idx_l0 < 0): use L1 MV/ref per spec
            let col_zero = if col_ref_l0 == 0 {
                let col_mv = if col_pos < col.mv_l0.len() {
                    col.mv_l0[col_pos]
                } else {
                    [0, 0]
                };
                col_mv[0].abs() <= 1 && col_mv[1].abs() <= 1
            } else if col_ref_l0 < 0
                && col_pos < col.ref_idx_l1.len()
                && col.ref_idx_l1[col_pos] == 0
            {
                let col_mv_l1 = if col_pos < col.mv_l1.len() {
                    col.mv_l1[col_pos]
                } else {
                    [0, 0]
                };
                col_mv_l1[0].abs() <= 1 && col_mv_l1[1].abs() <= 1
            } else {
                false
            };

            if col_zero {
                if ref_idx[0] == 0 {
                    mv[0] = [0, 0];
                }
                if ref_idx[1] == 0 {
                    mv[1] = [0, 0];
                }
            }
        }
    }

    (
        mv[0],
        mv[1],
        ref_idx[0],
        ref_idx[1],
        pred_flag[0],
        pred_flag[1],
    )
}

/// Derive temporal direct mode MVs for a specific 4x4 block within an MB.
/// Per spec 8.4.1.2.3, reads the co-located block's MV and scales by POC distance.
#[allow(clippy::type_complexity)]
pub(crate) fn derive_temporal_direct_blk(
    col_pic: &DecodedPicture,
    ref_pic_list_l0: &[Rc<DecodedPicture>],
    current_poc: i32,
    col_poc: i32,
    mb_idx: usize,
    blk: usize,
    direct_8x8_inference_flag: bool,
) -> ([i16; 2], [i16; 2], i8, i8, bool, bool) {
    // When direct_8x8_inference_flag is set (spec 8.4.1.2.3), use ONE co-located
    // MV per 8x8 block. Per spec, the co-located partition is derived using the
    // MV at luma4x4BlkIdx = (x8*3, y8*3) in 4x4-block raster coordinates within
    // the co-located MB. Mapping to our BLOCK_INDEX_TO_OFFSET numbering:
    //   8x8 blk 0 (x8=0,y8=0): raster(0,0) = pixel(0,0)   = our block 0
    //   8x8 blk 1 (x8=1,y8=0): raster(3,0) = pixel(0,12)  = our block 5
    //   8x8 blk 2 (x8=0,y8=1): raster(0,3) = pixel(12,0)  = our block 10
    //   8x8 blk 3 (x8=1,y8=1): raster(3,3) = pixel(12,12) = our block 15
    let col_blk = if direct_8x8_inference_flag {
        const INFERENCE_MAP: [usize; 4] = [0, 5, 10, 15];
        INFERENCE_MAP[blk / 4]
    } else {
        blk
    };
    let col_base = mb_idx * 16;
    if col_pic.is_intra
        || col_base + col_blk >= col_pic.ref_idx_l0.len()
        || col_pic.ref_idx_l0[col_base + col_blk] < 0
    {
        // Intra co-located: zero MVs, ref_idx=0
        return ([0, 0], [0, 0], 0, 0, true, true);
    }

    // Get co-located MV and ref_idx
    let col_mv = col_pic.mv_l0[col_base + col_blk];
    let col_ref_idx = col_pic.ref_idx_l0[col_base + col_blk];

    // Map co-located ref_idx to current L0 ref_idx by matching POC (spec 8.4.1.2.3).
    // Look up the POC that the co-located picture referenced at col_ref_idx.
    let col_ref_poc_val = col_pic.ref_poc_l0[col_base + col_blk];

    // Find the entry in our current L0 list with the matching POC.
    let ref0 = ref_pic_list_l0
        .iter()
        .position(|p| p.pic_order_cnt == col_ref_poc_val)
        .unwrap_or(0);
    let poc0 = ref_pic_list_l0
        .get(ref0)
        .map(|p| p.pic_order_cnt)
        .unwrap_or(0);

    // Compute dist_scale_factor: td = col_poc - poc0, tb = current_poc - poc0
    let td = (col_poc - poc0).clamp(-128, 127);
    let tb = (current_poc - poc0).clamp(-128, 127);

    let (mv_l0, mv_l1) = if td == 0 {
        // No scaling possible
        (col_mv, [0, 0])
    } else {
        let tx = (16384 + (td.abs() >> 1)) / td;
        let scale = ((tb * tx + 32) >> 6).clamp(-1024, 1023);
        let mx_l0 = ((scale * col_mv[0] as i32 + 128) >> 8) as i16;
        let my_l0 = ((scale * col_mv[1] as i32 + 128) >> 8) as i16;
        let mx_l1 = mx_l0 - col_mv[0];
        let my_l1 = my_l0 - col_mv[1];
        ([mx_l0, my_l0], [mx_l1, my_l1])
    };

    (mv_l0, mv_l1, ref0 as i8, 0, true, true)
}

/// Motion vector prediction using the median of neighbors A, B, C (spec 8.4.1.3).
/// `part_idx`: partition index (0 for first/only partition).
/// `part_w`, `part_h`: partition dimensions.
/// `ref_idx`: reference index for this partition.
#[allow(clippy::too_many_arguments)]
pub(crate) fn predict_mv(
    mv_store_l0: &[[i16; 2]],
    ref_idx_store_l0: &[i8],
    mb_idx: usize,
    mb_width: usize,
    part_idx: usize,
    part_w: usize,
    part_h: usize,
    ref_idx: i8,
    mb_slice_id: &[u16],
    cur_slice_id: u16,
) -> (i16, i16) {
    let py_off = if part_h == 8 && part_w == 16 {
        part_idx * 8
    } else {
        0
    };
    let px_off = if part_w == 8 && part_h == 16 {
        part_idx * 8
    } else {
        0
    };

    // A: left neighbor (4x4 block to the left of partition's top-left)
    let a = get_mv_neighbor_left(
        mv_store_l0,
        ref_idx_store_l0,
        mb_idx,
        mb_width,
        py_off,
        px_off,
        mb_slice_id,
        cur_slice_id,
    );

    // B: above neighbor
    let b = get_mv_neighbor_above(
        mv_store_l0,
        ref_idx_store_l0,
        mb_idx,
        mb_width,
        py_off,
        px_off,
        mb_slice_id,
        cur_slice_id,
    );

    // C: above-right neighbor (or D: above-left if C unavailable)
    let c = get_mv_neighbor_above_right(
        mv_store_l0,
        ref_idx_store_l0,
        mb_idx,
        mb_width,
        py_off,
        px_off,
        part_w,
        mb_slice_id,
        cur_slice_id,
    )
    .or_else(|| {
        get_mv_neighbor_above_left(
            mv_store_l0,
            ref_idx_store_l0,
            mb_idx,
            mb_width,
            py_off,
            px_off,
            mb_slice_id,
            cur_slice_id,
        )
    });

    // Special cases for 16x8 and 8x16 (spec 8.4.1.3.1)
    if part_w == 16 && part_h == 8 {
        if part_idx == 0 {
            if let Some((mv, ri)) = b {
                if ri == ref_idx {
                    return (mv[0], mv[1]);
                }
            }
        } else if let Some((mv, ri)) = a {
            if ri == ref_idx {
                return (mv[0], mv[1]);
            }
        }
    }
    if part_w == 8 && part_h == 16 {
        if part_idx == 0 {
            if let Some((mv, ri)) = a {
                if ri == ref_idx {
                    return (mv[0], mv[1]);
                }
            }
        } else if let Some((mv, ri)) = c {
            if ri == ref_idx {
                return (mv[0], mv[1]);
            }
        }
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
            if let Some((mv, _)) = a {
                return (mv[0], mv[1]);
            }
        }
        if ref_b == ref_idx {
            if let Some((mv, _)) = b {
                return (mv[0], mv[1]);
            }
        }
        if ref_c == ref_idx {
            if let Some((mv, _)) = c {
                return (mv[0], mv[1]);
            }
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
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn get_mv_neighbor_left(
    mv_store_l0: &[[i16; 2]],
    ref_idx_store_l0: &[i8],
    mb_idx: usize,
    mb_width: usize,
    py_off: usize,
    px_off: usize,
    mb_slice_id: &[u16],
    cur_slice_id: u16,
) -> Option<([i16; 2], i8)> {
    let mb_col = mb_idx % mb_width;
    if px_off > 0 {
        // Left is within this MB
        let lr = py_off / 4;
        let lc = (px_off - 4) / 4;
        let blk = OFFSET_TO_BLOCK[lr][lc];
        Some((
            mv_store_l0[mb_idx * 16 + blk],
            ref_idx_store_l0[mb_idx * 16 + blk],
        ))
    } else if mb_col > 0 {
        // Left is in the left MB (rightmost column)
        let left_mb = mb_idx - 1;
        if mb_slice_id[left_mb] != cur_slice_id {
            return None;
        }
        let lr = py_off / 4;
        let lc = 3; // rightmost 4x4 column
        let blk = OFFSET_TO_BLOCK[lr][lc];
        Some((
            mv_store_l0[left_mb * 16 + blk],
            ref_idx_store_l0[left_mb * 16 + blk],
        ))
    } else {
        None
    }
}

/// Get MV/ref of the above neighbor for a partition.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn get_mv_neighbor_above(
    mv_store_l0: &[[i16; 2]],
    ref_idx_store_l0: &[i8],
    mb_idx: usize,
    mb_width: usize,
    py_off: usize,
    px_off: usize,
    mb_slice_id: &[u16],
    cur_slice_id: u16,
) -> Option<([i16; 2], i8)> {
    let mb_row = mb_idx / mb_width;
    if py_off > 0 {
        // Above is within this MB
        let lr = (py_off - 4) / 4;
        let lc = px_off / 4;
        let blk = OFFSET_TO_BLOCK[lr][lc];
        Some((
            mv_store_l0[mb_idx * 16 + blk],
            ref_idx_store_l0[mb_idx * 16 + blk],
        ))
    } else if mb_row > 0 {
        let above_mb = mb_idx - mb_width;
        if mb_slice_id[above_mb] != cur_slice_id {
            return None;
        }
        let lr = 3; // bottom row
        let lc = px_off / 4;
        let blk = OFFSET_TO_BLOCK[lr][lc];
        Some((
            mv_store_l0[above_mb * 16 + blk],
            ref_idx_store_l0[above_mb * 16 + blk],
        ))
    } else {
        None
    }
}

/// Get MV/ref of the above-right neighbor for a partition.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn get_mv_neighbor_above_right(
    mv_store_l0: &[[i16; 2]],
    ref_idx_store_l0: &[i8],
    mb_idx: usize,
    mb_width: usize,
    py_off: usize,
    px_off: usize,
    part_w: usize,
    mb_slice_id: &[u16],
    cur_slice_id: u16,
) -> Option<([i16; 2], i8)> {
    let mb_col = mb_idx % mb_width;
    let mb_row = mb_idx / mb_width;
    let right_col = px_off + part_w;

    if py_off > 0 {
        // Above-right within this MB
        if right_col < 16 {
            // Per spec 6.4.11.7: the above-right 4x4 block is not available
            // if it would be in a different 8x8 block that hasn't been decoded.
            // Within P_8x8 / B_8x8, each 8x8 sub-MB is decoded independently.
            // The above-right at (py_off-4, px_off+part_w) is unavailable if
            // it crosses an 8x8 column boundary while staying in the same 8x8 row.
            let cur_8x8_col = px_off / 8;
            let tgt_col = right_col;
            let tgt_8x8_col = tgt_col / 8;
            // Also check row: above-right row is py_off-4
            let tgt_8x8_row = (py_off - 4) / 8;
            let cur_8x8_row = py_off / 8;
            let cur_8x8 = cur_8x8_row * 2 + cur_8x8_col;
            let tgt_8x8 = tgt_8x8_row * 2 + tgt_8x8_col;
            // Target must be in an already-decoded 8x8 block (lower index in scan)
            if tgt_8x8 > cur_8x8 {
                return None;
            }
            let lr = (py_off - 4) / 4;
            let lc = right_col / 4;
            let blk = OFFSET_TO_BLOCK[lr][lc];
            Some((
                mv_store_l0[mb_idx * 16 + blk],
                ref_idx_store_l0[mb_idx * 16 + blk],
            ))
        } else {
            None // right edge of MB, above-right is in MB above-right
        }
    } else if mb_row > 0 {
        // Above-right in the MB above (or above-right MB)
        if right_col < 16 {
            let above_mb = mb_idx - mb_width;
            if mb_slice_id[above_mb] != cur_slice_id {
                return None;
            }
            let lr = 3;
            let lc = right_col / 4;
            let blk = OFFSET_TO_BLOCK[lr][lc];
            Some((
                mv_store_l0[above_mb * 16 + blk],
                ref_idx_store_l0[above_mb * 16 + blk],
            ))
        } else if mb_col + 1 < mb_width {
            let above_right_mb = mb_idx - mb_width + 1;
            if mb_slice_id[above_right_mb] != cur_slice_id {
                return None;
            }
            let blk = OFFSET_TO_BLOCK[3][0];
            Some((
                mv_store_l0[above_right_mb * 16 + blk],
                ref_idx_store_l0[above_right_mb * 16 + blk],
            ))
        } else {
            None
        }
    } else {
        None
    }
}

/// Get MV/ref of the above-left neighbor for a partition (fallback for C).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn get_mv_neighbor_above_left(
    mv_store_l0: &[[i16; 2]],
    ref_idx_store_l0: &[i8],
    mb_idx: usize,
    mb_width: usize,
    py_off: usize,
    px_off: usize,
    mb_slice_id: &[u16],
    cur_slice_id: u16,
) -> Option<([i16; 2], i8)> {
    let mb_col = mb_idx % mb_width;
    let mb_row = mb_idx / mb_width;

    if py_off > 0 && px_off > 0 {
        let lr = (py_off - 4) / 4;
        let lc = (px_off - 4) / 4;
        let blk = OFFSET_TO_BLOCK[lr][lc];
        Some((
            mv_store_l0[mb_idx * 16 + blk],
            ref_idx_store_l0[mb_idx * 16 + blk],
        ))
    } else if py_off == 0 && px_off == 0 && mb_row > 0 && mb_col > 0 {
        // Above-left MB, bottom-right block
        let al_mb = mb_idx - mb_width - 1;
        if mb_slice_id[al_mb] != cur_slice_id {
            return None;
        }
        let blk = OFFSET_TO_BLOCK[3][3];
        Some((
            mv_store_l0[al_mb * 16 + blk],
            ref_idx_store_l0[al_mb * 16 + blk],
        ))
    } else if py_off == 0 && px_off > 0 && mb_row > 0 {
        let above_mb = mb_idx - mb_width;
        if mb_slice_id[above_mb] != cur_slice_id {
            return None;
        }
        let lc = (px_off - 4) / 4;
        let blk = OFFSET_TO_BLOCK[3][lc];
        Some((
            mv_store_l0[above_mb * 16 + blk],
            ref_idx_store_l0[above_mb * 16 + blk],
        ))
    } else if py_off > 0 && px_off == 0 && mb_col > 0 {
        let left_mb = mb_idx - 1;
        if mb_slice_id[left_mb] != cur_slice_id {
            return None;
        }
        let lr = (py_off - 4) / 4;
        let blk = OFFSET_TO_BLOCK[lr][3];
        Some((
            mv_store_l0[left_mb * 16 + blk],
            ref_idx_store_l0[left_mb * 16 + blk],
        ))
    } else {
        None
    }
}
