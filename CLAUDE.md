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

Intra-only I-frame decoding is functional. P/B slice decoding not yet implemented.

### Completed

**Slice Header Parsing** (`src/slice.rs`)
- Full slice header parsing: slice_type, frame_num, pic_order_cnt, slice_qp_delta
- Decoded reference picture marking for IDR slices
- Deblocking filter parameter parsing

**Macroblock Decoding** (`src/decoder.rs`)
- I4x4 macroblocks with all 9 prediction modes
- I16x16 macroblocks with all 4 prediction modes (vertical, horizontal, DC, plane)
- I_PCM macroblocks (raw pixel data)
- Coded Block Pattern (CBP) handling for luma and chroma
- Per-macroblock QP delta

**CAVLC Entropy Decoding** (`src/cavlc.rs`)
- Complete coeff_token VLC tables (nC 0-2, 2-4, 4-8, 8+, chroma DC)
- Trailing ones and level parsing with suffix length adaptation
- Total zeros and run-before VLC tables
- Zigzag scan order handling
- O(1) VLC decode via flat peek-indexed lookup tables (built once via `OnceLock`)

**NAL Unit Parsing** (`src/nal.rs`)
- Annex B start code detection (3-byte and 4-byte)
- Emulation prevention byte removal (`00 00 03` → `00 00`)
- forbidden_zero_bit validation — invalid NAL units silently skipped

**Bitstream Reader** (`src/bitstream.rs`)
- MSB-first bit reading with `read_bit`, `read_bits`, `read_ue`, `read_se`
- Non-consuming `peek_bits(n)` and position-advancing `skip_bits(n)`

**Intra Prediction** (`src/intra_pred.rs`)
- I16x16: vertical, horizontal, DC, plane (4 modes)
- I4x4: vertical, horizontal, DC, diagonal down-left/right, vertical-right/left, horizontal-down/up (9 modes)
- Chroma 8x8: DC, horizontal, vertical, plane (4 modes)
- Neighbor mode derivation with cross-macroblock support

**Transform & Quantization** (`src/residual.rs`)
- 4x4 inverse integer DCT
- 4x4 inverse Hadamard (I16x16 luma DC)
- 2x2 inverse Hadamard (chroma DC)
- Dequantization with H.264 LevelScale tables

**Deblocking Filter** (`src/deblock.rs`)
- H.264 spec section 8.7 loop filter
- Strong filter (bS=4) for MB boundary edges and normal filter (bS=3) for internal edges
- Luma and chroma filtering with per-edge QP-based threshold computation
- Alpha, beta, tc0 lookup tables from H.264 Tables 8-16a/b/c
- Applied automatically after slice decode; respects `disable_deblocking_filter_idc`

**Test Coverage**
- `testdata/single_frame.h264` - 16x16 I16x16 frame
- `testdata/multi_mb_frame.h264` - 64x64 multi-macroblock I16x16 frame
- `testdata/i4x4_frame.h264` - 16x16 I4x4 frame (deblocking disabled in stream)
- `testdata/deblock_frame.h264` - 64x64 I16x16 checkerboard pattern with deblocking enabled
- `testdata/mixed_i4x4_frame.h264` - 64x64 mixed I4x4/I16x16 checkerboard with deblocking
- `testdata/gradient_48x32.h264` - 48x32 (3x2 MBs) vertical gradient, QP=40
- `testdata/edges_32x32_qp10.h264` - 32x32 high-contrast bar pattern, QP=10
- `testdata/edges_32x32_qp35.h264` - 32x32 high-contrast bar pattern, QP=35
- `testdata/smooth_80x48.h264` - 80x48 (5x3 MBs) smooth gradient with colored chroma
- `testdata/noise_16x16_qp12.h264` - 16x16 single-MB pseudo-random content, QP=12
- All test outputs validated byte-for-byte against FFmpeg's decoder

### Not Yet Implemented

- P and B slice macroblock types (inter prediction)
- Motion compensation
- Reference picture buffer management
- MBAFF/interlaced mode
- Scaling lists
