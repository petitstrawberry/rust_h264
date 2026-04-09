# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-04-09

### Added
- `OrderedDecoder` — wraps `Decoder` with a built-in reorder buffer that
  emits frames in display order automatically. Handles IDR boundary tracking
  internally, eliminating a common pitfall where callers would tag the last
  B-frame of a GOP with the wrong GOP id and produce visible glitches at
  scene cuts. Recommended for most users.
- AVCC (length-prefixed) NAL parser for MP4/MKV containers:
  - `nal::parse_avcc(data, length_size)` — parse a length-prefixed sample
  - `nal::parse_avcc_config(avcc_box)` — parse an MP4 `avcC` configuration
    record and extract SPS/PPS NALs
  - `nal::AvccConfig` struct exposing `length_size`, `sps_nals`, `pps_nals`
- NEON SIMD acceleration on aarch64 for luma half-pel filters
  (`half_pel_h`, `half_pel_v`, quarter-pel paths) and chroma bilinear
  interpolation. Requires no opt-in — enabled automatically on aarch64.
- Comprehensive doc comments on the entire public API. Crate-level docs,
  module-level intros, and runnable examples render on docs.rs.
- Test for `OrderedDecoder` validating display-order output against the
  manual sort baseline on a real preset_medium stream.

### Changed
- **Performance**: 1080p decode improved from ~48 fps to ~67 fps (28% speedup)
  on Apple Silicon thanks to NEON luma half-pel intrinsics. 1080p @ 60 fps
  target achieved.
- **Performance**: 720p P-only improved from ~186 fps to ~278 fps;
  720p with B-frames from ~190 fps to ~299 fps.
- README.md restructured to lead with `OrderedDecoder` as the recommended
  entry point. The lower-level `Decoder` is documented as an advanced
  option for callers who need raw decode order.
- `parse_annex_b` and AVCC parsers now share a common `parse_nal_bytes`
  helper for NAL header parsing and emulation-prevention removal.

### Fixed
- Build clean on stable Rust with no clippy warnings.

## [0.2.0] - 2026-04-04

### Added
- Initial release on crates.io. Pure Rust H.264 decoder supporting Baseline,
  Main, and High profile (8-bit 4:2:0 progressive) with both CAVLC and CABAC
  entropy coding.
- Full support for I/P/B-frame decoding, multi-reference prediction with
  `ref_pic_list_modification`, multi-slice frames, weighted prediction
  (explicit and implicit), long-term references, MMCO ops 1-6, deblocking
  filter, intra prediction (I4x4/I8x8/I16x16, all 9 modes for 4x4 and 8x8),
  spatial and temporal direct mode for B-slices.
- Annex B bitstream parser (`nal::parse_annex_b`) with zero-copy emulation
  prevention removal.
- 119 unit tests, including 73 byte-exact stream tests against FFmpeg, with
  coverage up to 1080p (CABAC, CABAC+deblock, CAVLC) using SHA-256 hash
  comparison to keep test data small.
- License: dual MIT / Apache-2.0.

### Performance
- ~48 fps at 1080p (mandelbrot content, x264 `--preset medium` bframes=3 ref=2)
- ~186 fps at 720p P-only / ~190 fps at 720p with B-frames
- See `BENCHMARK.md` for detailed profile breakdown and methodology.

### Not implemented
- Interlaced coding (MBAFF, field pictures)
- High 10 / 4:2:2 / 4:4:4 profiles
- SP/SI slice types
- Slice groups / FMO
