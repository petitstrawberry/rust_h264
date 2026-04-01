//! Per-slice mutable decode state shared across all MB decode paths.
//!
//! `SliceContext` bundles the ~25 mutable arrays and geometry values that
//! every MB decoder (CABAC/CAVLC, I/P/B) reads and writes. Extracting it
//! from `decode_slice` enables splitting MB decode logic into methods.

use crate::decoder::Frame;

/// Mutable per-slice state passed to MB decode routines.
///
/// All mutable MB-level arrays live here so that individual decode
/// functions can be extracted as methods on `&mut SliceContext`.
#[allow(dead_code)] // Fields will be read in Phase 2 when MB decode moves to methods
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
