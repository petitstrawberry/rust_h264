# Benchmark: rust_h264 vs FFmpeg

## Test Setup

- **Platform:** Apple Silicon (ARM64), macOS
- **Stream:** 1280×720, 300 frames, x264 `--preset medium --bframes 0 --ref 1 --no-deblock`, CABAC
- **FFmpeg:** Single-threaded (`-threads 1`), software decode, compiled with `-O3` + NEON assembly
- **rust_h264:** `cargo build --release`, pure Rust, no SIMD

## Results

| Decoder | Time (user) | FPS | Memory |
|---------|-------------|-----|--------|
| FFmpeg (1 thread) | 0.01s | ~15,000 | 20 MB |
| rust_h264 (release) | 1.63s | 184 | 12 MB |

**FFmpeg is ~84× faster.** This is expected — FFmpeg has decades of hand-tuned NEON/SSE assembly for the hot paths.

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
the surrounding code in `decode_cabac_mb`: stores to MV/ref/MVD arrays,
`BLOCK_INDEX_TO_OFFSET` linear scans, and function call overhead.

**Approaches:**
- **`BLOCK_INDEX_TO_OFFSET` lookup table:** Currently uses `.iter().position()`
  (O(16) linear scan) called dozens of times per MB. Replace with a direct
  `[usize; 4][4]` table mapping `(row/4, col/4) → block_index`. This alone
  could save ~5% of CABAC overhead.
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
- **Strength reduction:** For full-pel chroma (frac=0), skip interpolation entirely
  and use `copy_from_slice`. Currently the code always runs the bilinear formula.

Expected improvement: **2-4× for chroma MC → ~5-10% overall**

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

## Known Decode Issues at 720p

The benchmark uses `--bframes 0 --ref 1` because some B-frame configurations
with `ref ≥ 2` produce pixel mismatches at late frames (frame 249+) in 720p
streams. This is likely the same B_8x8 B_Direct_8x8 sub-partition context
issue that was fixed for smaller resolutions but may have residual edge cases
at larger frame counts where error accumulates.

## Realistic Performance Target

A pure-Rust decoder without SIMD can realistically achieve **~500 fps at 720p**
(~3× current) through:
1. BLOCK_INDEX_TO_OFFSET lookup table (+5%)
2. Full-pel MC fast path (+10%)
3. Loop unrolling / batch MC processing (+30%)
4. Inline critical neighbor lookups (+5%)
5. Reduce redundant array stores (+5%)

For real-time 720p/30fps, the current 184 fps is already **6× realtime**.
For 1080p/30fps, SIMD would be necessary.
