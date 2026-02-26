# H.264 Decoder Implementation Plan

## Current State

I-frame decoding is partially functional:
- I16x16 and I_PCM macroblocks decode correctly
- I4x4 macroblocks fail for multi-coefficient blocks (see `I4x4_DECODING_BUG.md`)
- Deblocking filter is parsed but not applied
- P/B slices return an error

---

## Phase 1: Complete I-Frame Decoding

### 1.1 Fix I4x4 CAVLC Bug

The decoder parses coefficients that are mathematically inconsistent with FFmpeg's output
from the same bitstream. Every sub-component (CAVLC bit tracing, zigzag, dequant, IDCT)
has been verified individually. The next debug step:

**Action:** Build FFmpeg with debug printf in `libavcodec/h264_cavlc.c`
(`ff_h264_decode_block_residual`) to print the actual coefficients FFmpeg parses for
`testdata/i4x4_frame.h264`. Compare to our output. If FFmpeg reads different bits, the
issue is byte offset (RBSP start, emulation prevention, header parsing). If FFmpeg reads
the same bits but converts differently, the issue is in level/coeff ordering.

Alternative: write a standalone C program using libavcodec to dump decoded MB residuals.

**Files:** `src/cavlc.rs`, `src/decoder.rs:186-230`

**Test:** `cargo test test_decode_i4x4_frame`

### 1.2 Deblocking Filter (`src/deblock.rs` — new file)

H.264 spec section 8.7. Must be applied after all MBs in a slice are reconstructed.

**Luma:**
1. Compute boundary strength (bS) for each 4x4 block edge:
   - bS=4 if either block is intra and edge is a macroblock boundary
   - bS=3 if either block is intra (non-MB boundary)
   - bS=2 if either block has non-zero residual coefficients
   - bS=1 if MV difference ≥ 1 pel or reference index differs (P/B only)
   - bS=0 otherwise
2. For each edge with bS > 0, apply the luma filter:
   - Thresholds: α, β from Table 8-16 (indexed by `filterOffsetA`, `filterOffsetB`)
   - Strong filter (bS=4, small Δ): modify p0, p1, p2, q0, q1, q2
   - Normal filter (bS=1..3): clip-based modification of p0, p1, q0, q1

**Chroma:** Same boundary strength, different threshold table (Table 8-16 chroma),
4 samples per edge instead of 16.

**Integration:**
- Add `disable_deblocking_filter_idc` check from slice header
- Skip deblocking across slice boundaries if idc=2
- Run deblocking in `decode_slice` after all MB reconstruction

---

## Phase 2: P-Slice Decoding

### 2.1 Reference Picture Buffer (`src/dpb.rs` — new file)

The Decoded Picture Buffer holds reconstructed frames available as references.

```
pub struct DecodedPicture {
    pub frame: Frame,
    pub frame_num: u32,
    pub poc: i32,
    pub is_long_term: bool,
    pub long_term_frame_idx: Option<u32>,
}

pub struct Dpb {
    pictures: Vec<DecodedPicture>,
    max_num_ref_frames: usize,
}
```

Operations:
- `insert(pic)` — add picture, evict oldest short-term if at capacity (sliding window)
- `mark_as_used_for_ref(frame_num)` / `mark_unused(frame_num)`
- `build_ref_list_p()` — sort short-term by descending frame_num, append long-term
- `mmco(ops)` — apply Memory Management Control Operations for non-IDR ref marking

**Integration:** `Decoder` gets a `Dpb` field. After each decoded frame, insert into DPB.
IDR frame clears DPB first.

### 2.2 Extended Slice Header Parsing (`src/slice.rs`)

For P slices, parse after the existing fields:
- `num_ref_idx_active_override_flag` + `num_ref_idx_l0_active_minus1`
- `ref_pic_list_modification` (reordering ops)
- `dec_ref_pic_marking` for non-IDR (MMCO operations)

### 2.3 P-Slice Macroblock Layer (`src/decoder.rs`)

P-slice mb_type values (Table 7-13):

| mb_type | Name           | Partitions |
|---------|----------------|------------|
| 0       | P_L0_16x16     | 1×(16×16)  |
| 1       | P_L0_L0_16x8   | 2×(16×8)   |
| 2       | P_L0_L0_8x16   | 2×(8×16)   |
| 3       | P_8x8          | 4×(8×8)    |
| 4       | P_8x8ref0      | 4×(8×8), ref=0 |
| 5+      | I slices (same as I slice mb_types + 5 offset) |

P_Skip: signalled via `mb_skip_run` prefix (Exp-Golomb), not via mb_type.

For each partition:
1. Parse `ref_idx_l0` (if more than 1 reference)
2. Parse `mvd_l0` (Exp-Golomb signed, two components)

**mb_skip_run parsing:** Before each mb_type, read `mb_skip_run`. For each skipped MB:
- Derive MV using P_Skip rules (spec 8.4.1.1)
- Zero residual, use motion compensation only

### 2.4 Motion Vector Prediction and Derivation (`src/mc.rs` — new file)

**Median predictor (spec 8.4.1.3.1):**
```
mvpA = left neighbor partition MV (or zero if unavailable)
mvpB = above neighbor partition MV
mvpC = above-right, or above-left if above-right unavailable
mvp = median(mvpA, mvpB, mvpC)  (component-wise)
```

Special cases:
- Only one valid neighbor: use that neighbor directly
- P_Skip: derive `mvp` as above, MV = mvp if collocated MV is zero, else mvp

Store per-MB, per-partition MV and reference index for neighbor access.

### 2.5 Motion Compensation (`src/mc.rs`)

**Luma (6-tap Wiener filter for half-pel):**

The H.264 luma interpolation uses a 6-tap filter: `[-1, 5, 20, 20, 5, -1] / 32`

Quarter-pel positions derived by averaging adjacent half-pel and integer-pel samples.

For each 4×4 luma sub-block in the partition:
1. Compute integer-pel address from MV >> 2
2. Horizontal shift = MV & 3, vertical shift = (MV >> 16) & 3 (or however stored)
3. Apply appropriate interpolation based on fractional offsets

**Chroma (bilinear):**
```
pred = ((8-dx)(8-dy)*A + dx(8-dy)*B + (8-dx)dy*C + dx*dy*D + 32) >> 6
```
where A/B/C/D are the four surrounding integer-pel chroma samples.

**Boundary handling:** Use `clip(x, 0, width-1)` / `clip(y, 0, height-1)` for out-of-frame
reference access (repeat edge pixels).

---

## Phase 3: B-Slice Decoding

### 3.1 Reference Picture Lists for B Slices

Two lists: L0 (forward references, past POC), L1 (backward references, future POC).

```
fn build_ref_list_b(dpb: &Dpb, current_poc: i32) -> (Vec<&DecodedPicture>, Vec<&DecodedPicture>)
```

L0: short-term with POC < current_poc sorted by descending POC, then long-term ascending
L1: short-term with POC > current_poc sorted by ascending POC, then long-term ascending

Swap L0 and L1 if they are identical (spec 8.2.4.2.3).

### 3.2 B-Slice mb_type (Table 7-14)

Key types (mb_type 0..22 are inter, 23 = B_Skip handled via skip_run):

| mb_type | Name              |
|---------|-------------------|
| 0       | B_Direct_16x16    |
| 1       | B_L0_16x16        |
| 2       | B_L1_16x16        |
| 3       | B_Bi_16x16        |
| 4–8     | 16x8 variants     |
| 9–13    | 8x16 variants     |
| 14–21   | B_8x8 (sub-MB)    |
| 22+     | I types (offset)  |

### 3.3 Spatial Direct Mode (spec 8.4.1.2.2)

For B_Direct_16x16 and B_Skip:
1. Find co-located MB in the L1[0] reference frame
2. Extract co-located MV (mvCol) and reference index (refIdxCol)
3. Derive L0 and L1 MVs from mvCol using temporal scaling

### 3.4 Bi-Prediction MC

Average L0 and L1 predictions:
```
pred = (pred_L0 + pred_L1 + 1) >> 1
```

Weighted prediction (if `weighted_bipred_idc != 0`) uses explicit weights from
`pred_weight_table` in the slice header.

---

## Phase 4: CABAC (`src/cabac.rs` — new file)

Required for Main Profile and High Profile streams. Signalled by `entropy_coding_mode_flag`
in PPS.

### 4.1 Arithmetic Decoder

State: `codIRange` (9 bits), `codIOffset` (9 bits).

```
fn decode_decision(ctx: &mut Context, range: &mut u16, offset: &mut u16, bits: &mut BitstreamReader) -> u8
fn decode_bypass(range: &mut u16, offset: &mut u16, bits: &mut BitstreamReader) -> u8
fn decode_terminate(range: &mut u16, offset: &mut u16, bits: &mut BitstreamReader) -> u8
```

Context model: `pStateIdx` (0..63) + `valMPS` (0 or 1). Table 9-1 gives transition rules.

### 4.2 Context Initialization

Two initialization tables (Table 9-12 through 9-34):
- Table index chosen by `cabac_init_idc` (0..2) from slice header
- Initial state derived from `SliceQPY` using `m*QP/2 + n` formula

### 4.3 CABAC Syntax Elements

Replace CAVLC calls with CABAC equivalents:
- `mb_type`: Table 9-36 (I slice), Table 9-37 (P/B)
- `coded_block_pattern`: Table 9-40
- `mb_qp_delta`: Table 9-41
- Residual: `coded_block_flag`, `significant_coeff_flag`, `last_significant_coeff_flag`,
  `coeff_abs_level_minus1`, `coeff_sign_flag` (Tables 9-42 to 9-44)
- Motion: `ref_idx`, `mvd_l0/l1` (Tables 9-38 to 9-39)

---

## Phase 5: Remaining Spec Features

### 5.1 Scaling Lists

SPS can include custom 4×4 and 8×8 quantization scaling lists (seq_scaling_list_present_flag).
PPS can override them. Currently using flat (all-16) implicit scaling.

Parse in `src/sps.rs` and `src/pps.rs`, apply in `dequant_4x4_full` and `dequant_4x4_ac_raster`.

### 5.2 POC Computation (`src/poc.rs` — new file)

Three methods (pic_order_cnt_type 0, 1, 2):
- Type 0 (most common): `TopFieldOrderCnt = pic_order_cnt_lsb + poc_msb`; requires
  tracking `PrevPicOrderCntLsb` across frames
- Type 1: derived from frame_num with offset cycles
- Type 2: directly from frame_num

POC is needed for B-slice reference list ordering and direct mode.

### 5.3 Long-term Reference Management

Currently IDR sets `long_term_reference_flag` from slice header but DPB doesn't exist yet.
Implement MMCO operations 1–6 in `Dpb::mmco`.

---

## Phase 6: Performance (Milestone 3)

After correctness is established, profile and optimize.

### 6.1 SIMD (`src/simd/`)

- `inverse_dct_4x4`: 16 i32 ops → SSE2 / NEON
- Luma 6-tap filter: primary bottleneck for P/B frames
- Intra prediction: memset patterns, averaging — trivial SIMD wins
- Use `std::arch` or the `wide` crate; gate with `#[cfg(target_arch = "x86_64")]`

### 6.2 Parallelism

- Slice-level: independent slices can decode in parallel (rare in practice)
- Frame-level (wavefront / frame threading): requires careful DPB synchronization

### 6.3 Allocations

- Reuse `Frame` buffers across frames via pool: `DpbFrame { data: Box<[u8]>, ... }`
- Avoid per-MB heap allocations; use stack arrays (already done in current code)

### 6.4 Benchmark

```bash
cargo bench  # criterion benchmarks
# Compare with:
ffmpeg -benchmark -i video.h264 -f null -
```

Target: within 2× of FFmpeg's software decoder for HD content.

---

## Implementation Order

1. **Fix I4x4 bug** — unblock Milestone 1
2. **Deblocking filter** — complete I-frame correctness
3. **DPB + POC** — prerequisite for inter frames
4. **P-slice decoding** — enables most real-world content
5. **B-slice decoding** — enables full baseline/main profile
6. **CABAC** — enables Main/High profile
7. **Scaling lists, MMCO** — correctness completeness
8. **SIMD + profiling** — Milestone 3

---

## File Map

| File | Purpose |
|------|---------|
| `src/bitstream.rs` | Bit-level reader (Exp-Golomb, read_bits) |
| `src/nal.rs` | Annex B parsing, NAL unit extraction |
| `src/sps.rs` | SPS RBSP parsing |
| `src/pps.rs` | PPS RBSP parsing |
| `src/slice.rs` | Slice header parsing |
| `src/cavlc.rs` | CAVLC entropy decoding |
| `src/cabac.rs` | CABAC arithmetic decoder *(todo)* |
| `src/intra_pred.rs` | I4x4, I16x16, chroma intra prediction |
| `src/residual.rs` | IDCT, Hadamard, dequantization, zigzag |
| `src/mc.rs` | Motion compensation, MV prediction *(todo)* |
| `src/deblock.rs` | Deblocking filter *(todo)* |
| `src/dpb.rs` | Decoded Picture Buffer *(todo)* |
| `src/poc.rs` | POC computation *(todo)* |
| `src/decoder.rs` | Top-level decode loop |
