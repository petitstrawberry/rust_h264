//! CABAC (Context-Adaptive Binary Arithmetic Coding) decoder.
//!
//! Implements the binary arithmetic decoder and context state management
//! per H.264 spec 9.3. Used as an alternative to CAVLC for entropy decoding
//! in Main and High profiles.

/// Number of bits used for buffer operations (16-bit double-byte mode).
const CABAC_BITS: u32 = 16;
/// Mask for buffer alignment checks.
const CABAC_MASK: u32 = (1 << CABAC_BITS) - 1;



#[rustfmt::skip]
static NORM_SHIFT: [u8; 512] = [
    9,8,7,7,6,6,6,6,5,5,5,5,5,5,5,5,
    4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
    3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
    3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
    2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
    2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
    2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
    2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
];

#[rustfmt::skip]
static LPS_RANGE: [u8; 512] = [
    // Range group 0
    128, 128, 128, 128, 128, 128, 123, 123,
    116, 116, 111, 111, 105, 105, 100, 100,
     95,  95,  90,  90,  85,  85,  81,  81,
     77,  77,  73,  73,  69,  69,  66,  66,
     62,  62,  59,  59,  56,  56,  53,  53,
     51,  51,  48,  48,  46,  46,  43,  43,
     41,  41,  39,  39,  37,  37,  35,  35,
     33,  33,  32,  32,  30,  30,  29,  29,
     27,  27,  26,  26,  24,  24,  23,  23,
     22,  22,  21,  21,  20,  20,  19,  19,
     18,  18,  17,  17,  16,  16,  15,  15,
     14,  14,  14,  14,  13,  13,  12,  12,
     12,  12,  11,  11,  11,  11,  10,  10,
     10,  10,   9,   9,   9,   9,   8,   8,
      8,   8,   7,   7,   7,   7,   7,   7,
      6,   6,   6,   6,   6,   6,   2,   2,
    // Range group 1
    176, 176, 167, 167, 158, 158, 150, 150,
    142, 142, 135, 135, 128, 128, 122, 122,
    116, 116, 110, 110, 104, 104,  99,  99,
     94,  94,  89,  89,  85,  85,  80,  80,
     76,  76,  72,  72,  69,  69,  65,  65,
     62,  62,  59,  59,  56,  56,  53,  53,
     50,  50,  48,  48,  45,  45,  43,  43,
     41,  41,  39,  39,  37,  37,  35,  35,
     33,  33,  31,  31,  30,  30,  28,  28,
     27,  27,  26,  26,  24,  24,  23,  23,
     22,  22,  21,  21,  20,  20,  19,  19,
     18,  18,  17,  17,  16,  16,  15,  15,
     14,  14,  14,  14,  13,  13,  12,  12,
     12,  12,  11,  11,  11,  11,  10,  10,
      9,   9,   9,   9,   9,   9,   8,   8,
      8,   8,   7,   7,   7,   7,   2,   2,
    // Range group 2
    208, 208, 197, 197, 187, 187, 178, 178,
    169, 169, 160, 160, 152, 152, 144, 144,
    137, 137, 130, 130, 123, 123, 117, 117,
    111, 111, 105, 105, 100, 100,  95,  95,
     90,  90,  86,  86,  81,  81,  77,  77,
     73,  73,  69,  69,  66,  66,  63,  63,
     59,  59,  56,  56,  54,  54,  51,  51,
     48,  48,  46,  46,  43,  43,  41,  41,
     39,  39,  37,  37,  35,  35,  33,  33,
     32,  32,  30,  30,  29,  29,  27,  27,
     26,  26,  25,  25,  23,  23,  22,  22,
     21,  21,  20,  20,  19,  19,  18,  18,
     17,  17,  16,  16,  15,  15,  15,  15,
     14,  14,  13,  13,  12,  12,  12,  12,
     11,  11,  11,  11,  10,  10,  10,  10,
      9,   9,   9,   9,   8,   8,   2,   2,
    // Range group 3
    240, 240, 227, 227, 216, 216, 205, 205,
    195, 195, 185, 185, 175, 175, 166, 166,
    158, 158, 150, 150, 142, 142, 135, 135,
    128, 128, 122, 122, 116, 116, 110, 110,
    104, 104,  99,  99,  94,  94,  89,  89,
     85,  85,  80,  80,  76,  76,  72,  72,
     69,  69,  65,  65,  62,  62,  59,  59,
     56,  56,  53,  53,  50,  50,  48,  48,
     45,  45,  43,  43,  41,  41,  39,  39,
     37,  37,  35,  35,  33,  33,  31,  31,
     30,  30,  28,  28,  27,  27,  25,  25,
     24,  24,  23,  23,  22,  22,  21,  21,
     20,  20,  19,  19,  18,  18,  17,  17,
     16,  16,  15,  15,  14,  14,  14,  14,
     13,  13,  12,  12,  12,  12,  11,  11,
     11,  11,  10,  10,   9,   9,   2,   2,
];

#[rustfmt::skip]
static MLPS_STATE: [u8; 256] = [
    // MPS transitions
    127, 126,  77,  76,  77,  76,  75,  74,
     75,  74,  75,  74,  73,  72,  73,  72,
     73,  72,  71,  70,  71,  70,  71,  70,
     69,  68,  69,  68,  67,  66,  67,  66,
     67,  66,  65,  64,  65,  64,  63,  62,
     61,  60,  61,  60,  61,  60,  59,  58,
     59,  58,  57,  56,  55,  54,  55,  54,
     53,  52,  53,  52,  51,  50,  49,  48,
     49,  48,  47,  46,  45,  44,  45,  44,
     43,  42,  43,  42,  39,  38,  39,  38,
     37,  36,  37,  36,  33,  32,  33,  32,
     31,  30,  31,  30,  27,  26,  27,  26,
     25,  24,  23,  22,  23,  22,  19,  18,
     19,  18,  17,  16,  15,  14,  13,  12,
     11,  10,   9,   8,   9,   8,   5,   4,
      5,   4,   3,   2,   1,   0,   0,   1,
    // LPS transitions
      2,   3,   4,   5,   6,   7,   8,   9,
     10,  11,  12,  13,  14,  15,  16,  17,
     18,  19,  20,  21,  22,  23,  24,  25,
     26,  27,  28,  29,  30,  31,  32,  33,
     34,  35,  36,  37,  38,  39,  40,  41,
     42,  43,  44,  45,  46,  47,  48,  49,
     50,  51,  52,  53,  54,  55,  56,  57,
     58,  59,  60,  61,  62,  63,  64,  65,
     66,  67,  68,  69,  70,  71,  72,  73,
     74,  75,  76,  77,  78,  79,  80,  81,
     82,  83,  84,  85,  86,  87,  88,  89,
     90,  91,  92,  93,  94,  95,  96,  97,
     98,  99, 100, 101, 102, 103, 104, 105,
    106, 107, 108, 109, 110, 111, 112, 113,
    114, 115, 116, 117, 118, 119, 120, 121,
    122, 123, 124, 125, 124, 125, 126, 127,
];

#[inline]
fn norm_shift(range: u32) -> u32 {
    NORM_SHIFT[range as usize] as u32
}

#[inline]
fn lps_range_lookup(range: u32, state: u8) -> u32 {
    LPS_RANGE[2 * (range & 0xC0) as usize + state as usize] as u32
}

#[inline]
fn mlps_state(s: u8) -> u8 {
    MLPS_STATE[128 + s as usize]
}

/// CABAC binary arithmetic decoder context.
pub struct CabacReader<'a> {
    low: u32,
    range: u32,
    data: &'a [u8],
    pos: usize,
}

impl<'a> CabacReader<'a> {
    /// Initialize the CABAC decoder from RBSP data at a given byte position.
    pub fn new(data: &'a [u8], byte_offset: usize) -> Self {
        let mut pos = byte_offset;
        let mut low: u32 = (data[pos] as u32) << 18;
        pos += 1;
        low += (data[pos] as u32) << 10;
        pos += 1;
        // Alignment: if next read is on 2-byte boundary, add 1<<9
        // otherwise read another byte
        if pos.is_multiple_of(2) {
            low += 1 << 9;
        } else {
            low += (data[pos] as u32) << 2;
            low += 2;
            pos += 1;
        }
        CabacReader {
            low,
            range: 0x1FE,
            data,
            pos,
        }
    }

    /// Refill the low register with 2 bytes from the bitstream.
    #[inline]
    fn refill(&mut self) {
        let b0 = if self.pos < self.data.len() { self.data[self.pos] } else { 0 };
        let b1 = if self.pos + 1 < self.data.len() { self.data[self.pos + 1] } else { 0 };
        self.low += (b0 as u32) << 9;
        self.low += (b1 as u32) << 1;
        self.low = self.low.wrapping_sub(CABAC_MASK);
        self.pos += 2;
    }

    /// Refill variant used after renormalization in get_cabac.
    #[inline]
    fn refill2(&mut self) {
        // Count trailing zeros to determine shift
        let i = self.low.trailing_zeros().wrapping_sub(CABAC_BITS);

        let b0 = if self.pos < self.data.len() { self.data[self.pos] } else { 0 };
        let b1 = if self.pos + 1 < self.data.len() { self.data[self.pos + 1] } else { 0 };
        let x = (b0 as u32) << 9 | (b1 as u32) << 1;
        let x = x.wrapping_sub(CABAC_MASK);
        self.low = self.low.wrapping_add(x << i);
        self.pos += 2;
    }

    /// Decode a single binary decision using the given context state.
    /// Returns the decoded bit (0 or 1) and updates the context state.
    #[inline]
    pub fn get_cabac(&mut self, state: &mut u8) -> u32 {
        let s = *state;
        let range_lps = lps_range_lookup(self.range, s);

        self.range -= range_lps;
        // Check if we're decoding the LPS
        let lps_mask = ((self.range << (CABAC_BITS + 1)).wrapping_sub(self.low)) >> 31;

        self.low = self.low.wrapping_sub((self.range << (CABAC_BITS + 1)) & lps_mask);
        self.range += (range_lps.wrapping_sub(self.range)) & lps_mask;

        let s = s ^ (lps_mask as u8);
        *state = mlps_state(s);
        let bit = (s & 1) as u32;

        // Renormalization
        let shift = norm_shift(self.range);
        self.range <<= shift;
        self.low <<= shift;
        if self.low & CABAC_MASK == 0 {
            self.refill2();
        }
        bit
    }

    /// Decode a bypass (equiprobable) bit — no context adaptation.
    #[inline]
    pub fn get_cabac_bypass(&mut self) -> u32 {
        self.low += self.low;
        if self.low & CABAC_MASK == 0 {
            self.refill();
        }
        let range = self.range << (CABAC_BITS + 1);
        if self.low < range {
            0
        } else {
            self.low -= range;
            1
        }
    }

    /// Decode a bypass bit and apply sign to the given value.
    /// Returns +val or -val.
    #[inline]
    pub fn get_cabac_bypass_sign(&mut self, val: i32) -> i32 {
        self.low += self.low;
        if self.low & CABAC_MASK == 0 {
            self.refill();
        }
        let range = self.range << (CABAC_BITS + 1);
        self.low = self.low.wrapping_sub(range);
        let mask = (self.low as i32) >> 31; // -1 if low >= range, 0 otherwise
        self.low = self.low.wrapping_add(range & mask as u32);
        (val ^ mask) - mask
    }

    /// Decode the end-of-slice flag.
    /// Returns 0 if more data, non-zero (bytes consumed) at end of slice.
    pub fn get_cabac_terminate(&mut self) -> u32 {
        self.range -= 2;
        if self.low < self.range << (CABAC_BITS + 1) {
            // Renormalize once
            let shift = (self.range.wrapping_sub(0x100)) >> 31;
            self.range <<= shift;
            self.low <<= shift;
            if self.low & CABAC_MASK == 0 {
                self.refill();
            }
            0
        } else {
            self.pos as u32
        }
    }
}

/// Initialize CABAC context states for a slice (spec 9.3.1.1).
/// Returns array of 1024 context state values initialized from QP and slice type.
pub fn init_cabac_states(slice_qp: i32, is_i_slice: bool, cabac_init_idc: u32) -> [u8; 1024] {
    let qp = slice_qp.clamp(0, 51);
    let tab: &[[i8; 2]; 1024] = if is_i_slice {
        &CABAC_CONTEXT_INIT_I
    } else {
        &CABAC_CONTEXT_INIT_PB[cabac_init_idc as usize]
    };

    let mut states = [0u8; 1024];
    for i in 0..1024 {
        let m = tab[i][0] as i32;
        let n = tab[i][1] as i32;
        let pre = 2 * ((m * qp) >> 4) + n - 127;
        let pre = pre.clamp(1, 126);
        // pre is now 1..126; state = pre (odd = MPS=1, even = MPS=0)
        states[i] = pre as u8;
    }
    states
}

use crate::cabac_tables::{CABAC_CONTEXT_INIT_I, CABAC_CONTEXT_INIT_PB};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_table_sizes() {
        assert_eq!(NORM_SHIFT.len(), 512);
        assert_eq!(LPS_RANGE.len(), 512);
        assert_eq!(MLPS_STATE.len(), 256);
    }

    #[test]
    fn test_norm_shift() {
        assert_eq!(norm_shift(0), 9);
        assert_eq!(norm_shift(1), 8);
        assert_eq!(norm_shift(128), 1);
        assert_eq!(norm_shift(255), 1);
        assert_eq!(norm_shift(256), 0);
    }

    #[test]
    fn test_lps_range_bounds() {
        // State 0 should give largest LPS range
        assert_eq!(lps_range_lookup(0x00, 0), 128);
        assert_eq!(lps_range_lookup(0x40, 0), 176);
        assert_eq!(lps_range_lookup(0x80, 0), 208);
        assert_eq!(lps_range_lookup(0xC0, 0), 240);
        // State 126 should give smallest LPS range (2)
        assert_eq!(lps_range_lookup(0x00, 126), 2);
        assert_eq!(lps_range_lookup(0xC0, 126), 2);
    }

    #[test]
    fn test_mlps_state_transitions() {
        // LPS transition from state 0 should go to state 2
        assert_eq!(mlps_state(0), 2);
        // MPS transition from state 126 should go to state 126
        assert_eq!(mlps_state(126), 126);
        assert_eq!(mlps_state(127), 127);
    }

    #[test]
    fn test_init_cabac_states() {
        // Just verify it doesn't panic and produces valid ranges
        let states = init_cabac_states(26, true, 0);
        for &s in &states {
            assert!(s >= 1 && s <= 126);
        }
    }
}
