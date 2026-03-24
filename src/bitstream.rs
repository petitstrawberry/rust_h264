/// Bitstream reader for parsing H.264 NAL unit RBSP data.
/// Reads bits left-to-right (MSB first) from a byte slice.
pub struct BitstreamReader<'a> {
    data: &'a [u8],
    byte_offset: usize,
    bit_offset: u8, // 0-7, bits already consumed in current byte
}

impl<'a> BitstreamReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_offset: 0,
            bit_offset: 0,
        }
    }

    /// Read a single bit, returning 0 or 1.
    pub fn read_bit(&mut self) -> Result<u8, &'static str> {
        if self.byte_offset >= self.data.len() {
            return Err("end of bitstream");
        }
        let bit = (self.data[self.byte_offset] >> (7 - self.bit_offset)) & 1;
        self.bit_offset += 1;
        if self.bit_offset == 8 {
            self.bit_offset = 0;
            self.byte_offset += 1;
        }
        Ok(bit)
    }

    /// Read `n` bits (up to 32) as a u32.
    pub fn read_bits(&mut self, n: u8) -> Result<u32, &'static str> {
        let mut val: u32 = 0;
        for _ in 0..n {
            val = (val << 1) | self.read_bit()? as u32;
        }
        Ok(val)
    }

    /// Peek at the next `n` bits without consuming them. Zero-pads if fewer bits remain.
    pub fn peek_bits(&self, n: u8) -> u32 {
        let mut val: u32 = 0;
        let mut byte_offset = self.byte_offset;
        let mut bit_offset = self.bit_offset;
        for _ in 0..n {
            if byte_offset < self.data.len() {
                val = (val << 1) | ((self.data[byte_offset] >> (7 - bit_offset)) & 1) as u32;
            } else {
                val <<= 1;
            }
            bit_offset += 1;
            if bit_offset == 8 {
                bit_offset = 0;
                byte_offset += 1;
            }
        }
        val
    }

    /// Advance position by `n` bits without reading them.
    pub fn skip_bits(&mut self, n: u8) {
        let total = self.bit_offset as usize + n as usize;
        self.byte_offset += total / 8;
        self.bit_offset = (total % 8) as u8;
    }

    /// Read an unsigned Exp-Golomb coded value (ue(v)).
    pub fn read_ue(&mut self) -> Result<u32, &'static str> {
        let mut leading_zeros: u32 = 0;
        while self.read_bit()? == 0 {
            leading_zeros += 1;
            if leading_zeros > 31 {
                return Err("exp-golomb overflow");
            }
        }
        if leading_zeros == 0 {
            return Ok(0);
        }
        let suffix = self.read_bits(leading_zeros as u8)?;
        Ok((1 << leading_zeros) - 1 + suffix)
    }

    /// Read a signed Exp-Golomb coded value (se(v)).
    pub fn read_se(&mut self) -> Result<i32, &'static str> {
        let code = self.read_ue()?;
        let val = code.div_ceil(2) as i32;
        if code % 2 == 0 {
            Ok(-val)
        } else {
            Ok(val)
        }
    }

    /// Returns true if there is more RBSP data before the trailing bits.
    /// The RBSP trailing bits are: a stop bit (1) followed by zero bits to byte-align.
    pub fn more_rbsp_data(&self) -> bool {
        if self.byte_offset >= self.data.len() {
            return false;
        }

        // Find the position of the last non-zero byte
        let mut last_nz = self.data.len();
        while last_nz > self.byte_offset && self.data[last_nz - 1] == 0 {
            last_nz -= 1;
        }
        if last_nz <= self.byte_offset {
            return false;
        }

        // If we're before the last non-zero byte, there's definitely more data
        if self.byte_offset < last_nz - 1 {
            return true;
        }

        // We're in the last non-zero byte. The RBSP stop bit is the lowest set bit.
        // Everything above it (towards MSB) is data. Check if there are unconsumed
        // data bits above the stop bit.
        let byte = self.data[self.byte_offset];
        // Trailing bits pattern: the stop bit is the least significant '1' bit,
        // with all bits below it being zero (byte alignment).
        // e.g., 0b1000_0000 is just a stop bit, 0b1010_0000 has 1 data bit + stop + 5 zeros.
        let stop_bit = byte.trailing_zeros(); // number of trailing zero bits
        // Bits in this byte that are data (not trailing): 8 - stop_bit - 1 (the stop bit itself)
        let data_bits_in_byte = 7 - stop_bit as u8;
        // There's more data if we haven't consumed all data bits yet
        self.bit_offset < data_bits_in_byte
    }

    /// Advance to the next byte boundary.
    pub fn align_to_byte(&mut self) {
        if self.bit_offset != 0 {
            self.bit_offset = 0;
            self.byte_offset += 1;
        }
    }

    pub fn position(&self) -> (usize, u8) {
        (self.byte_offset, self.bit_offset)
    }

    /// Peek at the byte at the current byte_offset (for debugging).
    pub fn peek_byte(&self) -> u8 {
        if self.byte_offset < self.data.len() {
            self.data[self.byte_offset]
        } else {
            0
        }
    }

    /// Peek at a slice of bytes starting at byte_offset (for debugging).
    pub fn peek_bytes(&self, n: usize) -> &[u8] {
        let end = (self.byte_offset + n).min(self.data.len());
        &self.data[self.byte_offset..end]
    }

    pub fn bits_remaining(&self) -> usize {
        if self.byte_offset >= self.data.len() {
            return 0;
        }
        (self.data.len() - self.byte_offset) * 8 - self.bit_offset as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_bits() {
        let data = [0b1010_0011, 0b1100_0000];
        let mut r = BitstreamReader::new(&data);
        assert_eq!(r.read_bits(4).unwrap(), 0b1010);
        assert_eq!(r.read_bits(4).unwrap(), 0b0011);
        assert_eq!(r.read_bits(2).unwrap(), 0b11);
    }

    #[test]
    fn test_read_ue() {
        // ue(0) = 1 -> code 0
        // ue(1) = 010 -> code 1
        // ue(2) = 011 -> code 2
        // ue(3) = 00100 -> code 3
        // Bit pattern: 1 010 011 00100 = 1010 0110 0100 = 0xA64
        let data = [0xA6, 0x40];
        let mut r = BitstreamReader::new(&data);
        assert_eq!(r.read_ue().unwrap(), 0);
        assert_eq!(r.read_ue().unwrap(), 1);
        assert_eq!(r.read_ue().unwrap(), 2);
        assert_eq!(r.read_ue().unwrap(), 3);
    }

    #[test]
    fn test_read_se() {
        // se mapping: code 0->0, 1->1, 2->-1, 3->2, 4->-2
        // ue values:   0       1       2       3       4
        // bits:        1      010     011    00100   00101
        // Combined: 1 010 011 00100 00101 = 1010 0110 0100 0010 1
        let data = [0xA6, 0x42, 0x80];
        let mut r = BitstreamReader::new(&data);
        assert_eq!(r.read_se().unwrap(), 0);
        assert_eq!(r.read_se().unwrap(), 1);
        assert_eq!(r.read_se().unwrap(), -1);
        assert_eq!(r.read_se().unwrap(), 2);
        assert_eq!(r.read_se().unwrap(), -2);
    }
}
