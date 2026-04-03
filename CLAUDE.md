# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A pure Rust H.264 video decoder library. Aims to be a standalone, portable software H.264 decoder (unlike OpenH264 which only supports baseline profile, or FFmpeg's decoder which isn't available as a separate library). Part of the broader rust_media ecosystem.

## Build Commands

This is a Rust project using Cargo:

- **Build:** `cargo build`
- **Test:** `cargo test`
- **Run single test:** `cargo test <test_name>`
- **Lint:** `cargo clippy`
- **Format:** `cargo fmt`
- **Check:** `cargo check`

## Milestones

1. Get simple decoder test case working
2. Finish implementation of decoder
3. Compare performance of decoder against ffmpeg

## Design Decisions

- **Input format:** Annex B bytestream (start code delimited), not AVCC (length-prefixed). Callers must provide raw Annex B NAL units.
- **Streaming API:** The decoder API is streaming — callers feed NAL units incrementally and receive decoded frames as they become available. No requirement to buffer an entire stream upfront.
- **Performance:** The decoder should be fast. Prefer efficient algorithms, minimize allocations, and avoid unnecessary copies. Performance relative to ffmpeg's software decoder is a key benchmark.

## Code Structure

The decoder logic is split across several files for maintainability:

| File | Lines | Content |
|------|-------|---------|
| `src/decoder.rs` | ~1,370 | Core `Decoder` struct, `decode_nal`, `decode_slice` MB loop, DPB/frame management |
| `src/decode_cabac.rs` | ~3,600 | CABAC MB decode: skip detection, mb_type dispatch, residual decode |
| `src/decode_cavlc.rs` | ~2,070 | CAVLC MB decode: P/B inter, intra, residual decode |
| `src/slice_context.rs` | ~920 | `SliceContext`/`SliceParams` structs + shared methods (skip, direct, reconstruct) |
| `src/mv_pred.rs` | ~940 | MV prediction, spatial/temporal direct mode derivation |
| `src/neighbor.rs` | ~470 | CABAC neighbor context helpers, nC computation, dequant helpers |

`SliceContext` bundles ~25 mutable per-MB arrays; `SliceParams` bundles read-only slice-level parameters. The `make_ctx!()` macro in `decoder.rs` constructs a `SliceContext` from local variables at each call site for zero-cost method dispatch.

## Status

I-frame, P-frame, and B-frame decoding fully functional with both CAVLC and CABAC. High profile 8x8 transform supported for both CAVLC and CABAC (intra and inter). Multi-reference (ref>1) with ref_pic_list_modification supported. Multi-slice frames fully supported for both CABAC and CAVLC (I, P, and B-frames byte-exact). All 49 test streams byte-exact against FFmpeg, including x264 `--preset medium` with and without deblocking (320x240, 60 frames, ref=4, bframes=3), and multi-slice streams with up to 4 slices per frame. Explicit weighted prediction for P-slices and B-slices, plus implicit weighted bi-prediction for B-slices.

### Completed

**Slice Header Parsing** (`src/slice.rs`)
- Full slice header parsing: slice_type, frame_num, pic_order_cnt, slice_qp_delta
- Decoded reference picture marking for IDR and non-IDR slices
- Deblocking filter parameter parsing
- P-slice fields: num_ref_idx_l0_active, ref_pic_list_modification, dec_ref_pic_marking
- B-slice fields: num_ref_idx_l1_active, direct_spatial_mv_pred_flag, L1 ref_pic_list_modification, pred_weight_table (consumed)

**Intra Macroblock Decoding** (`src/decode_cabac.rs`, `src/decode_cavlc.rs`)
- I4x4 macroblocks with all 9 prediction modes
- I16x16 macroblocks with all 4 prediction modes (vertical, horizontal, DC, plane)
- I_PCM macroblocks (raw pixel data, both CAVLC and CABAC with engine reinit)
- Coded Block Pattern (CBP) handling for luma and chroma
- Per-macroblock QP delta
- Intra MBs within P-slices

**Inter Macroblock Decoding** (`src/decode_cabac.rs`, `src/decode_cavlc.rs`)
- P_Skip macroblocks (MV = median predictor, no residual)
- P_L0_16x16 (single 16x16 partition with ref_idx, MVD, residual)
- P_L0_L0_16x8 and P_L0_L0_8x16 (two-partition modes)
- MV prediction with median and directional (match_count) logic
- Inter CBP table, inter scaling lists (indices 3-5)
- P_8x8 with all sub-partition types (8x8, 8x4, 4x8, 4x4) and P_8x8ref0
- B_Skip (spatial/temporal direct mode, per-4x4-block MV derivation, no residual)
- B_Direct_16x16 (spatial/temporal direct mode, per-4x4-block MV derivation + residual)
- B_L0_16x16, B_L1_16x16 (uni-directional), B_Bi_16x16 (bi-directional)
- Dual MV/ref_idx storage (L0 + L1) for B-slice support
- Spatial direct mode: min-positive ref_idx from neighbors, median MV prediction,
  per-4x4-block co-located zero-MV refinement
- Temporal direct mode: per-4x4-block co-located MV scaling by POC distance (dist_scale_factor),
  `direct_8x8_inference_flag` support (one MV per 8x8 group from co-located picture)
- Bi-prediction averaging for luma and chroma

**Motion Compensation** (`src/inter_pred.rs`)
- Luma: 6-tap FIR filter for half-pel, bilinear averaging for quarter-pel (all 16 positions per spec Table 8-12)
- Chroma: bilinear interpolation at eighth-pel precision
- Bi-prediction: `bi_pred_avg` pixel averaging of L0 and L1 predictions
- Weighted prediction: `weighted_uni` (explicit P/B), `weighted_bi` (explicit B),
  `weighted_bi_implicit` (implicit B with POC-distance weights)
- Boundary clipping per spec 8.4.2.2.1

**Decoded Picture Buffer** (`src/dpb.rs`)
- Reference frame storage with `Rc<DecodedPicture>` sharing
- Sliding window marking (spec 8.2.5.3)
- POC computation for types 0, 1, 2
- P-slice L0: short-term refs sorted by descending frame_num
- B-slice L0: refs sorted by POC (before current descending, after ascending)
- B-slice L1: refs sorted by POC (after current ascending, before descending)
- `ref_pic_list_modification` (spec 8.2.4.3) with shift+insert+dedup algorithm,
  including idc=2 for long-term reference reordering
- Long-term reference support: MMCO ops 1-6 (mark ST/LT unused, assign ST→LT,
  set max LT index, clear all, assign current as LT), IDR `long_term_reference_flag`
- Long-term refs appended to reference lists after short-term refs (spec 8.2.4.2)
- Co-located picture MV/ref storage for temporal direct mode

**CAVLC Entropy Decoding** (`src/cavlc.rs`)
- Complete coeff_token VLC tables (nC 0-2, 2-4, 4-8, 8+, chroma DC)
- Trailing ones and level parsing with suffix length adaptation
- Total zeros and run-before VLC tables
- O(1) VLC decode via flat peek-indexed lookup tables (built once via `OnceLock`)

**CABAC Entropy Decoding** (`src/cabac.rs`, `src/cabac_tables.rs`)
- Binary arithmetic decoder: `get_cabac`, `get_cabac_bypass`, `get_cabac_terminate`
- Context state initialization from QP with 1024 contexts (I-slice + 3 P/B variants)
- Syntax element decoders: mb_type, skip, CBP, pred modes, ref_idx, MVD, sub_mb_type, QP delta
- Residual coefficient decoder: significance map + coefficient levels with 8-node state machine
- I4x4 and I16x16 integration: byte-exact output for single-MB, ±1 IDCT tolerance for multi-MB
- Per-MB neighbor tracking: CBF (luma LEFT[16]/TOP[16] + chroma), CBP (u16 with DC coded flags),
  chroma pred mode, I16x16 flag — all with proper unavailable-intra defaults (0x7CF)
- P-slice CABAC: P_Skip, P_L0_16x16/16x8/8x16, P_8x8 (all sub-partition types), intra-in-P
- B-slice CABAC: B_Skip (spatial/temporal direct), B_Direct_16x16, B_L0/L1/Bi_16x16,
  B 16x8/8x16 (18 partition variants), B_8x8 (13 sub_mb_types including B_Direct_8x8),
  intra-in-B (I4x4 and I16x16)
- Dual MVD stores (L0 + L1) for B-slice CABAC amvd context
- Category 5 (8x8 luma): no coded_block_flag (CBP bit sufficient), per-position context offsets
- `transform_size_8x8_flag` context: `399 + neighbor_transform_size` with `mb_is_8x8dct` tracking

**NAL Unit Parsing** (`src/nal.rs`)
- Annex B start code detection (3-byte and 4-byte)
- Emulation prevention byte removal with zero-copy fast path (`Cow::Borrowed`)
- forbidden_zero_bit validation

**Bitstream Reader** (`src/bitstream.rs`)
- MSB-first bit reading with `read_bit`, `read_bits`, `read_ue`, `read_se`, `read_te`
- Non-consuming `peek_bits(n)` and position-advancing `skip_bits(n)`
- Padded buffer for bounds-check-free `read_bit`

**Intra Prediction** (`src/intra_pred.rs`)
- I16x16: vertical, horizontal, DC, plane (4 modes)
- I4x4: all 9 modes with above-right availability checks
- I8x8: all 9 modes at 8×8 granularity with low-pass filtered reference samples
- Chroma 8x8: DC (per-4x4-quadrant), horizontal, vertical, plane (4 modes)

**Transform & Quantization** (`src/residual.rs`)
- 4x4 inverse integer DCT
- 8x8 inverse integer DCT (High profile)
- 4x4 inverse Hadamard (I16x16 luma DC)
- 2x2 inverse Hadamard (chroma DC)
- Dequantization with 4x4 and 8x8 scaling list support (SPS/PPS, fallback to default matrices)

**Deblocking Filter** (`src/deblock.rs`)
- Strong filter (bS=4) and normal filter (bS=1-3) with proper luma/chroma distinction
- Chroma: strong filter modifies only p0/q0 (spec 8.7.2.4); normal filter uses tc=tc0+1 (spec 8.7.2.3)
- Full spec 8.7.2.1 per-4x4-block boundary strength derivation:
  bS=4 (intra MB edge), bS=3 (intra internal), bS=2 (non-zero coefficients),
  bS=1 (different refs or |MV_diff|>=4), bS=0 (none). B-slice dual-list
  straight+swapped comparison.
- 8x8 transform: internal odd edges (positions 4 and 12 within MB) skipped
  per spec 8.7.2.1 — they fall inside 8x8 transform blocks
- Applied automatically after slice decode

**Multi-Slice Frame Support** (`src/decoder.rs`, `src/decode_cabac.rs`, `src/decode_cavlc.rs`)
- `PictureState` accumulates decoded MBs across multiple slices of the same picture
- Frame finalization (deblocking, DPB insert) on next picture's first slice or `flush()`
- `mb_slice_id` array tracks which slice each MB belongs to
- All CABAC neighbor context functions check slice boundaries (skip, mb_type, CBP,
  chroma pred, 8x8dct, ref_idx, MVD, coded_block_flag)
- Intra prediction sample availability gated on same-slice membership (spec 6.4.1):
  cross-slice neighbors treated as unavailable for I4x4, I8x8, I16x16 luma and chroma
  prediction in all code paths (CABAC I-slice, CABAC intra-in-P/B, CAVLC I-slice,
  CAVLC intra-in-P/B)
- CAVLC `compute_nc` checks slice boundaries for cross-MB nC derivation
- CAVLC continuation slice error recovery: backup/restore of `PictureState`

**Error Handling** (`src/error.rs`)
- `DecodeError` enum with `UnexpectedEof`, `InvalidSyntax`, `Unsupported` variants
- Prediction functions use graceful fallback instead of panicking

**Test Coverage** (115 tests, all byte-exact against FFmpeg)
- Intra (CAVLC): single_frame, multi_mb_frame, i4x4_frame, deblock_frame,
  mixed_i4x4_frame, gradient_48x32, edges (QP=10/35), smooth_80x48,
  noise_16x16, scaling_test
- P-slice: p_frame_test, p_skip_heavy, p_multi_frame, p_8x8_test, p_multiref
- B-slice: b_l0_l1_test, b_bi_test, b_skip_test (spatial direct),
  b_temporal_test, b_parts_test (16x8/8x16/8x8), b_multi_test, b_hier_test
  (hierarchical B-frames with ref_pic_list_modification)
- CABAC: cabac_i4x4_test, cabac_i16x16_test, cabac_mixed_test,
  cabac_p_test, cabac_p_parts_test (P16x8/8x16/8x8/4x4 sub-partitions),
  cabac_intra_p_test (I16x16-in-P),
  cabac_b_test (B_Skip spatial direct),
  cabac_b_parts_test (B16x16/16x8/8x16/8x8/Direct/Skip with L0/L1/Bi),
  cabac_intra_b_test (I16x16-in-B), cabac_b_temporal_test (temporal direct),
  cabac_high_profile (8x8 inter), cabac_deblock_test (deblocking enabled),
  cabac_i8x8_test (I8x8 intra with chroma),
  cabac_multiref_test (ref=2 with ref_pic_list_modification)
- Deblocking: deblock_frame, deblock_b_test (B-frames + deblock),
  deblock_b_inter_test (B inter + cross-list bS)
- Weighted prediction: weighted_p_test (CAVLC, 100% weighted P, fading),
  weighted_b_test (CABAC, implicit weighted B idc=2)
- High profile: high_profile_test (320x240 CAVLC, 8x8 intra+inter)
- Real-world: realworld_test (320x240 P-only),
  realworld_b_test (320x240 with B-frames),
  preset_medium (320x240, 60 frames, x264 --preset medium, no-deblock),
  preset_medium_deblock (same with deblocking ON)
- Multi-slice: ms_cabac_i_test (32x32, 2-slice CABAC I-frame),
  ms_cabac_i4_test (64x64, 4-slice CABAC I-frame),
  ms_cavlc_i_test (32x32, 2-slice CAVLC I-frame),
  ms_cavlc_p_test (64x64, 5-frame 4-slice CAVLC with P-frames),
  ms_cabac_p_test (64x64, 5-frame 4-slice CABAC with P-frames),
  ms_cabac_b_test (64x64, 4-frame 4-slice CABAC with B-frames),
  b_temporal_direct_test (64x64, 4-frame, preset slower temporal direct 8x8 inference),
  high_p8x8_sub4x4_test (64x64, 6-frame, High profile P_8x8 sub-4x4 + 8x8dct),
  high_b_slower_test (64x64, 10-frame, High profile preset slower ref=2 B-frames 8x8dct),
  high_cavlc_b_test (64x64, 10-frame, CAVLC High profile bframes=2 ref=2 8x8dct),
  ms_deblock_b_cabac_test (64x64, 8-frame, CABAC 4-slice bframes=2 ref=2 deblock),
  ms_cavlc_b_test (64x64, 8-frame, CAVLC 4-slice bframes=2 ref=2),
  cavlc_deblock_pb_test (64x64, 8-frame, CAVLC P+B with deblocking),
  unaligned_100x76_test (100x76, 6-frame, non-16-aligned dimensions),
  cabac_weighted_p_test (64x64, 8-frame, CABAC 100% weighted P fading),
  cavlc_i8x8_test (64x64, 3-frame, CAVLC High profile I8x8 intra),
  high_preset_medium_test (320x240, 30-frame, High profile bframes=3 ref=4 8x8dct no-deblock),
  high_deblock_medium_test (320x240, 30-frame, High profile bframes=3 ref=4 8x8dct deblock ON),
  constrained_intra_test (64x64, 8-frame, CABAC constrained_intra_pred_flag=1 bframes=1 ref=2),
  cabac_b8x8_direct_test (64x64, 8-frame, CABAC B_8x8 with B_Direct_8x8 sub-partitions),
  jm_ltr_cavlc_test (64x64, 8-frame, JM CAVLC long-term reference),
  jm_ltr_cabac_test (64x64, 8-frame, JM CABAC long-term reference),
  jm_weighted_b_explicit_test (64x64, 8-frame, JM CABAC weighted_bipred_idc=1),
  jm_poc_type1_test (64x64, 6-frame, JM CAVLC pic_order_cnt_type=1),
  jm_poc_type2_test (64x64, 6-frame, JM CAVLC pic_order_cnt_type=2),
  jm_ipcm_cavlc_test (32x32, 4-frame, JM CAVLC I_PCM macroblocks QP=0)

### Not Yet Implemented

- MBAFF/interlaced mode (mb_adaptive_frame_field, field pictures)
- High 10/4:2:2/4:4:4 profiles (>8-bit, non-4:2:0 chroma)
- SP/SI slice types (parsed but not decoded)
- Slice groups / FMO (returns error)
