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

## Status

I-frame, P-frame, and B-frame decoding functional. B_Skip, B_Direct_16x16, B_L0_16x16, B_L1_16x16, and B_Bi_16x16 implemented (spatial direct mode). B-slice 16x8/8x16/8x8 sub-partitions and temporal direct mode not yet implemented.

### Completed

**Slice Header Parsing** (`src/slice.rs`)
- Full slice header parsing: slice_type, frame_num, pic_order_cnt, slice_qp_delta
- Decoded reference picture marking for IDR and non-IDR slices
- Deblocking filter parameter parsing
- P-slice fields: num_ref_idx_l0_active, ref_pic_list_modification, dec_ref_pic_marking
- B-slice fields: num_ref_idx_l1_active, direct_spatial_mv_pred_flag, L1 ref_pic_list_modification, pred_weight_table (consumed)

**Intra Macroblock Decoding** (`src/decoder.rs`)
- I4x4 macroblocks with all 9 prediction modes
- I16x16 macroblocks with all 4 prediction modes (vertical, horizontal, DC, plane)
- I_PCM macroblocks (raw pixel data)
- Coded Block Pattern (CBP) handling for luma and chroma
- Per-macroblock QP delta
- Intra MBs within P-slices

**Inter Macroblock Decoding** (`src/decoder.rs`)
- P_Skip macroblocks (MV = median predictor, no residual)
- P_L0_16x16 (single 16x16 partition with ref_idx, MVD, residual)
- P_L0_L0_16x8 and P_L0_L0_8x16 (two-partition modes)
- MV prediction with median and directional (match_count) logic
- Inter CBP table, inter scaling lists (indices 3-5)
- P_8x8 with all sub-partition types (8x8, 8x4, 4x8, 4x4) and P_8x8ref0
- B_Skip (spatial direct mode, no residual)
- B_Direct_16x16 (spatial direct mode + residual)
- B_L0_16x16, B_L1_16x16 (uni-directional), B_Bi_16x16 (bi-directional)
- Dual MV/ref_idx storage (L0 + L1) for B-slice support
- Spatial direct mode: min-positive ref_idx from neighbors, median MV prediction
- Bi-prediction averaging for luma and chroma

**Motion Compensation** (`src/inter_pred.rs`)
- Luma: 6-tap FIR filter for half-pel, bilinear averaging for quarter-pel (all 16 positions)
- Chroma: bilinear interpolation at eighth-pel precision
- Bi-prediction: `bi_pred_avg` pixel averaging of L0 and L1 predictions
- Boundary clipping per spec 8.4.2.2.1

**Decoded Picture Buffer** (`src/dpb.rs`)
- Reference frame storage with `Rc<DecodedPicture>` sharing
- Sliding window marking (spec 8.2.5.3)
- POC computation for types 0, 1, 2
- P-slice L0: short-term refs sorted by descending frame_num
- B-slice L0: refs sorted by POC (before current descending, after ascending)
- B-slice L1: refs sorted by POC (after current ascending, before descending)

**CAVLC Entropy Decoding** (`src/cavlc.rs`)
- Complete coeff_token VLC tables (nC 0-2, 2-4, 4-8, 8+, chroma DC)
- Trailing ones and level parsing with suffix length adaptation
- Total zeros and run-before VLC tables
- O(1) VLC decode via flat peek-indexed lookup tables (built once via `OnceLock`)

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
- Chroma 8x8: DC (per-4x4-quadrant), horizontal, vertical, plane (4 modes)

**Transform & Quantization** (`src/residual.rs`)
- 4x4 inverse integer DCT
- 4x4 inverse Hadamard (I16x16 luma DC)
- 2x2 inverse Hadamard (chroma DC)
- Dequantization with scaling list support (SPS/PPS, fallback to default matrices)

**Deblocking Filter** (`src/deblock.rs`)
- Strong filter (bS=4) and normal filter (bS=1-3)
- Inter-aware boundary strength derivation
- Applied automatically after slice decode

**Error Handling** (`src/error.rs`)
- `DecodeError` enum with `UnexpectedEof`, `InvalidSyntax`, `Unsupported` variants
- Prediction functions use graceful fallback instead of panicking

**Test Coverage** (56 tests)
- Intra: single_frame, multi_mb_frame, i4x4_frame, deblock_frame, mixed_i4x4_frame,
  gradient_48x32, edges (QP=10/35), smooth_80x48, noise_16x16, scaling_test
- P-slice: p_frame_test (IDR+P), p_skip_heavy (50% skip), p_multi_frame (IDR+3P
  with P16x16/P16x8/8x16/intra-in-P), p_8x8_test (82.8% P_8x8 + sub-8x4),
  p_multiref (IDR+3P with ref=3, multi-reference P8x16)
- B-slice: b_l0_l1_test (100% B_L0_16x16), b_bi_test (33% B_Bi + 67% B_L1 +
  intra-in-B), b_skip_test (100% B_Skip, spatial direct mode)

### Not Yet Implemented

- Temporal direct mode (co-located MV scaling)
- Co-located zero-MV refinement for spatial direct mode
- B-slice 16x8, 8x16, B_8x8 partitions
- Multi-B-frame sequences (consecutive B-frames)
- MBAFF/interlaced mode
- CABAC entropy decoding
