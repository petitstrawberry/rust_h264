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

**Test Coverage**
- `testdata/single_frame.h264` - 16x16 I16x16 frame
- `testdata/multi_mb_frame.h264` - 64x64 multi-macroblock frame
- `testdata/i4x4_frame.h264` - 16x16 I4x4 frame

### Not Yet Implemented

- P and B slice macroblock types (inter prediction)
- Motion compensation
- Deblocking filter (parameters parsed but filtering not applied)
- Reference picture buffer management
- MBAFF/interlaced mode
- Scaling lists
