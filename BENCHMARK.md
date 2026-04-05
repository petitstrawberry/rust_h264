# Benchmark: rust_h264 vs FFmpeg

## Test Setup

- **Platform:** Apple Silicon (ARM64), macOS
- **Streams:** 1280×720, 300 frames, x264 `--preset medium --no-deblock`, CABAC
  - P-only: `--bframes 0 --ref 1`
  - B-frames: `--bframes 1 --ref 1 --no-weightb`
- **FFmpeg:** Single-threaded (`-threads 1`), software decode, compiled with `-O3` + NEON assembly
- **rust_h264:** `cargo build --release`, pure Rust, no SIMD

## Results

| Decoder | Stream | Time (user) | FPS | Memory |
|---------|--------|-------------|-----|--------|
| FFmpeg | P-only | 0.01s | ~30,000 | 20 MB |
| FFmpeg | B-frames | 0.01s | ~30,000 | 20 MB |
| rust_h264 | P-only | 1.61s | 186 | 12 MB |
| rust_h264 | B-frames | 1.37s | 190 | 12 MB |

**FFmpeg is ~80-160× faster.** This is expected — FFmpeg has decades of hand-tuned
NEON/SSE assembly for the hot paths.

## Profile Breakdown

Sampled with macOS `sample` command on the 720p P-only decode:

| Component | % Time | Description |
|-----------|--------|-------------|
| **Luma MC (half-pel filters)** | **42%** | 6-tap FIR filter for sub-pixel interpolation |
| **CABAC decode overhead** | **25%** | Loop/store overhead (18.7%), arithmetic engine (2.8%), residual/cbp/mvd (3.5%) |
| **Chroma MC** | **18%** | Bilinear interpolation at 1/8-pel precision |
| **P_Skip MC** | **10%** | Combined luma+chroma MC for skip MBs |
| Inverse DCT | 3% | 4×4 integer IDCT |
| finalize_mb_info | 2% | Per-MB metadata copy for deblocking |
| Other | 1% | MV prediction, reconstruction, etc. |

### B-frame profile (720p, bframes=3 ref=4, CABAC)

Sampled with macOS `sample` on 300-frame 720p B-frame decode (1.40s user):

| Component | % Time | Description |
|-----------|--------|-------------|
| **Luma MC** | **42%** | 6-tap FIR half-pel filters (25% in B_Skip, 16% in other inter) |
| **Chroma MC** | **19%** | Bilinear 1/8-pel (13% B_Skip, 6% other inter) |
| **Spatial direct MV** | **9%** | `derive_spatial_direct_blk` per-4x4-block derivation |
| **Bi-pred averaging** | **7%** | L0+L1 pixel averaging in B_Skip |
| **CABAC decode** | **5%** | Residual (2%), syntax elements (2%), neighbor/dequant (1%) |
| Deblock/frame mgmt | 4% | Deblocking filter + DPB management |
| Reconstruct | 2% | Luma/chroma reconstruction from residual |
| Inverse DCT | 2% | 4×4 integer IDCT |
| Other | 10% | MV prediction, malloc, unaccounted |

**Key difference from P-only:** B_Skip dominates (55% of total), with spatial
direct MV derivation (9%) as a new significant cost. CABAC overhead dropped
from 25% to 5% after the `OFFSET_TO_BLOCK` optimization — the reverse lookups
were a major B-slice bottleneck since each B MB required dual-list neighbor
queries.

## Optimization Opportunities

### 1. Luma MC half-pel filters (42%) — High impact, medium effort

The 6-tap FIR filter (`half_pel_h`, `half_pel_v`, `half_pel_hv`) dominates.
Each output pixel requires 6 multiplications + additions + clipping.

**Approaches:**
- **SIMD (NEON):** Process 8 or 16 pixels per instruction. FFmpeg achieves ~8× speedup
  with NEON for these filters. Rust supports NEON via `std::arch::aarch64` intrinsics
  or the `packed_simd` / `std::simd` (nightly) crate.
- **Batch processing:** Current code processes one pixel at a time in a loop.
  Restructuring to process entire rows would improve cache locality.
- **Pre-computed filter tables:** For common block sizes (16×16, 8×8), unrolled
  filter kernels avoid loop overhead.

Expected improvement: **3-5× for MC alone → ~1.5-2× overall**

### 2. CABAC decode (25%) — Medium impact, medium effort

The CABAC arithmetic engine (`get_cabac`) is only 2.8% — the actual bottleneck is
the surrounding code in `decode_cabac_mb`: stores to MV/ref/MVD arrays and
function call overhead.

**Approaches:**
- ~~**`BLOCK_INDEX_TO_OFFSET` lookup table:**~~ Done — see optimization #6 below.
- **Inline `cabac_neighbor_*` functions:** The neighbor context lookups involve
  multiple function calls with many parameters. `#[inline(always)]` or manual
  inlining would reduce call overhead.
- **Reduce array stores:** MV/ref/MVD stores write to every 4×4 block (16 writes
  for a 16×16 partition). For uniform partitions, a single `memset`-style fill
  would be faster.

Expected improvement: **~10-20% of CABAC time → ~3-5% overall**

### 3. Chroma MC (18%) — Medium impact, low effort

Chroma MC uses bilinear interpolation at 1/8-pel. Simpler than luma but still
per-pixel with multiplications.

**Approaches:**
- **SIMD:** Same NEON approach as luma MC.
- ~~**Strength reduction:** For full-pel chroma (frac=0), skip interpolation entirely
  and use `copy_from_slice`.~~ Done — see optimization #7 below.

Expected improvement: **2-4× for chroma MC → ~5-10% overall** (SIMD only; full-pel
fast path already implemented)

### 4. Inverse DCT (3%) — Low impact

Already fast. SIMD could help for 8×8 IDCT in High profile but the 4×4 IDCT
is simple enough that scalar code is nearly optimal.

### 5. Memory allocation (done)

Replaced `Vec` heap allocations with stack arrays in hot paths:
- MC prediction buffers: `vec![0u8; w*h]` → `[0u8; 256]`
- `b_sub_parts`: `Vec<BSubPart>` → `[BSubPart; 16]`
- `BSubLayout`: `Vec<BSubLayout>` → `[BSubLayout; 4]`
- Sub-partition offsets: `vec![...]` → `&[...]` static slices

**Result: ~4% improvement** (1.70s → 1.63s)

### 6. OFFSET_TO_BLOCK reverse lookup table (done)

Replaced ~46 O(16) linear scans (`BLOCK_INDEX_TO_OFFSET.iter().position()`) with
O(1) `OFFSET_TO_BLOCK[row][col]` table lookups across `neighbor.rs`,
`decode_cabac.rs`, `decode_cavlc.rs`, and `mv_pred.rs`. These reverse lookups
convert (row, col) grid coordinates to block indices and were called dozens of
times per MB for neighbor context (amvd, ref_idx, coded_block_flag) and MV
prediction.

**Result: ~11% improvement on B-frames** (1.58s → 1.40s), P-only within noise
(1.60s → 1.65s). The B-frame gain is larger because B-slices exercise the
reverse lookup much more heavily: dual-list neighbor lookups, direct mode checks,
and spatial/temporal MV derivation.

### 7. Full-pel MC fast path (done)

Added early-exit fast paths in `luma_mc` and `chroma_mc`: when the fractional MV
is zero (integer-pel position), skip the 6-tap FIR / bilinear interpolation and
`copy_from_slice` directly from the reference buffer. Inner-bounds check avoids
per-pixel clamping for blocks fully within the picture.

**Result: ~2% improvement** (P-only 1.65s → 1.61s, B-frames 1.40s → 1.37s).
Modest because x264 `--preset medium` (subme=7) produces mostly sub-pel MVs.
Streams with simpler motion estimation or static content would see larger gains.

## Known Issues

**Frame ordering bug (fixed):** The `dump_frames` example had a frame
reordering bug at IDR boundaries — `idr_count` was incremented when the
IDR NAL was seen (before `decode_nal`), but `decode_nal` returns the
PREVIOUS frame. This caused the last B-frame of the first GOP to be
tagged with the second GOP's IDR count, placing it after the second
IDR's frames in display order. Fixed by incrementing `idr_count` after
`decode_nal` returns.

**Spatial direct colZeroFlag L1 fallback (fixed):** At 720p with
`ref=4 bframes=3` and smooth sinusoidal content, ±1 pixel diffs
appeared in B-frames using spatial direct mode. Root cause: per spec
8.4.1.2.2, when the co-located partition is L1-only (`PredFlagL0=0`),
`mvCol`/`refIdxCol` should be derived from L1 data, not L0. Our code
only stored and checked L0 data from the co-located picture. For L1-only
co-located blocks (`ref_idx_l0 < 0`), we missed the colZeroFlag entirely.
Fixed by storing L1 MV/ref data in `DecodedPicture` and using L1 data
when L0 is unavailable in `derive_spatial_direct_blk`.

## Realistic Performance Target

A pure-Rust decoder without SIMD can realistically achieve **~500 fps at 720p**
(~3× current) through:
1. ~~BLOCK_INDEX_TO_OFFSET lookup table (+5%)~~ Done — ~11% B-frame improvement
2. ~~Full-pel MC fast path (+10%)~~ Done — ~2% (content-dependent)
3. Loop unrolling / batch MC processing (+30%)
4. Inline critical neighbor lookups (+5%)
5. Reduce redundant array stores (+5%)

For real-time 720p/30fps, the current ~185 fps is already **6× realtime**.
For 1080p/30fps, SIMD would be necessary.
