# I4x4 Decoding Bug

## Summary

The I4x4 macroblock decoding produces incorrect output for test cases with multiple non-zero coefficients, while simpler cases (DC-only) and I16x16 macroblocks decode correctly.

## Failing Test

`test_decode_i4x4_frame` - Decodes `testdata/i4x4_frame.h264` and compares against `testdata/i4x4_frame.yuv` (ffmpeg reference output).

### Expected vs Actual Output (Block 0, Row 0)
- **Expected pixels**: `[16, 16, 81, 80]`
- **Actual pixels**: `[95, 81, 95, 132]`

### Test File Details
- File: `testdata/i4x4_frame.h264`
- MD5: `916e890962cf5e5c3dd55599a9b47c46`
- Encoder: x264 core 165 (cabac=0, 8x8dct=0)
- Profile: Constrained Baseline
- Resolution: 16x16 (single macroblock)

## What Works

| Test | Description | Status |
|------|-------------|--------|
| I16x16 single frame | IDR with I16x16 macroblock | ✅ Pass |
| multi_mb_frame | Multiple macroblocks | ✅ Pass |
| Solid grey (128) | All pixels = 128, zero coefficients | ✅ Pass |
| Solid color (64) | All pixels = 64, DC-only coefficient | ✅ Pass |
| Simple pattern | Left=32, right=224, DC-only per block | ✅ Pass |

## What Fails

| Test | Description | Status |
|------|-------------|--------|
| i4x4_frame | Complex pattern, 14 coefficients in block 0 | ❌ Fail |
| gradient | Horizontal gradient pattern | ❌ Fail |

## Components Verified as Correct

### 1. Bit-Level CAVLC Parsing
Manually traced bits from position 71 (byte 8, bit 7) matching decoder output:
- coeff_token: 16 bits for (14, 0)
- 14 levels parsed correctly
- total_zeros = 0
- run_before = all zeros

### 2. Level Values
Parsed levels match manual bit tracing:
```
[-19, 20, 20, -6, -16, -6, -10, -10, 30, 16, 51, 16, -51, -140]
```

### 3. Coefficient Placement
After reversing and applying run_before:
```
Scan order: [-140, -51, 16, 51, 16, 30, -10, -10, -6, -16, -6, 20, 20, -19, 0, 0]
```

### 4. Zigzag Scan Table
Verified against H.264 Figure 8-8:
```rust
const ZIGZAG_4X4: [(usize, usize); 16] = [
    (0, 0), (0, 1), (1, 0), (2, 0),
    (1, 1), (0, 2), (0, 3), (1, 2),
    (2, 1), (3, 0), (3, 1), (2, 2),
    (1, 3), (2, 3), (3, 2), (3, 3),
];
```

### 5. Dequantization
For QP=7 (qp_per=1, qp_rem=1), LevelScale[1] = [11, 18, 14]:
- DC at (0,0): -140 × 11 × 2 = -3080 ✅
- Position (0,1): -51 × 14 × 2 = -1428 ✅

### 6. IDCT
Verified step-by-step against H.264 spec 8.5.12:
- Horizontal pass: correct intermediate values
- Vertical pass with (x + 32) >> 6 rounding: matches output

### 7. Prediction Mode
Block 0 uses DC mode (2) with no neighbors → prediction = 128

## The Core Mystery

### What We Get
- Parsed DC level: -140
- Dequantized DC: -3080
- IDCT output row 0: `[-33, -47, -33, 4]`
- Final pixels: `[95, 81, 95, 132]`

### What We Need
- Expected residuals: `[-112, -112, -47, -48]`
- Forward DCT of expected: DC ≈ -1007
- Required quantized DC: ≈ -46 (not -140)

### The Discrepancy
The coefficients in the bitstream (-140 DC) are completely different from what would produce the expected output (-46 DC). Yet:
1. FFmpeg decodes the same bitstream correctly
2. Our bit-level parsing matches the actual bits
3. Every individual component (dequant, IDCT, prediction) is mathematically correct

## Hypotheses Explored

### 1. Emulation Prevention Bytes
Checked for 0x000003 sequences - none found in the slice data.

### 2. Wrong QP
Verified: pic_init_qp_minus26 = -16, slice_qp_delta = -3, QP_Y = 7

### 3. Wrong MB Type
Verified: mb_type = 0 → I_NxN (I4x4 for baseline profile)

### 4. Position Mismatch
Reader position (8, 7) = bit 71 matches manual calculation.

### 5. Different Zigzag for Field Mode
Baseline profile only supports frame mode; our zigzag table is correct.

## Key Difference: I4x4 vs I16x16

| Aspect | I4x4 | I16x16 |
|--------|------|--------|
| DC handling | Part of 16-coeff block | Separate 4x4 Hadamard |
| Coefficients | All 16 together | DC extracted, AC separate |
| Status | ❌ Fails (multi-coeff) | ✅ Works |

## Files Referenced

- `src/cavlc.rs` - CAVLC parsing (level conversion fixed)
- `src/decoder.rs` - I4x4 decoding path (lines 116-275)
- `src/residual.rs` - ZIGZAG_4X4, dequant_4x4_full, inverse_dct_4x4

## Suggested Next Steps

1. **Compare with FFmpeg source**: Trace FFmpeg's h264_cavlc.c to see exact coefficient values
2. **Binary search**: Create test files with varying coefficient counts to find threshold
3. **Check for transform bypass**: Some modes skip IDCT (unlikely for baseline)
4. **Verify intra prediction derivation**: Double-check neighbor mode prediction algorithm
5. **Test with other encoders**: Try encoding with different H.264 encoders

## Test Files Created During Debug

These were removed but can be recreated:
- `grey16.{h264,yuv}` - solid 128 grey
- `solid64.{h264,yuv}` - solid 64
- `gradient.{h264,yuv}` - horizontal gradient
- `simple_pattern.{h264,yuv}` - left/right pattern

## Related Fix

The level conversion formula was fixed during this investigation:
```rust
// Old (buggy):
if suffix_length == 0 {
    if level_code & 1 == 0 { -level_val - 1 } else { level_val }
}

// New (correct):
let level_val = (level_code + 2) >> 1;
let level = if level_code & 1 != 0 { -level_val } else { level_val };
```

This fixed I16x16 decoding but did not resolve the I4x4 multi-coefficient issue.
