use rust_h264::nal::{parse_annex_b, NalUnitType};

// Table 9-5(a): 0 <= nC < 2
#[rustfmt::skip]
static COEFF_TOKEN_NC0: [(u32, u8, u8, u8); 62] = [
    (0b1, 1, 0, 0), (0b000101, 6, 1, 0), (0b01, 2, 1, 1), (0b00000111, 8, 2, 0),
    (0b000100, 6, 2, 1), (0b001, 3, 2, 2), (0b000000111, 9, 3, 0), (0b00000110, 8, 3, 1),
    (0b0000101, 7, 3, 2), (0b00011, 5, 3, 3), (0b0000000111, 10, 4, 0), (0b000000110, 9, 4, 1),
    (0b00000101, 8, 4, 2), (0b000011, 6, 4, 3), (0b00000000111, 11, 5, 0), (0b0000000110, 10, 5, 1),
    (0b000000101, 9, 5, 2), (0b0000100, 7, 5, 3), (0b0000000001111, 13, 6, 0), (0b00000000110, 11, 6, 1),
    (0b0000000101, 10, 6, 2), (0b00001000, 8, 6, 3), (0b0000000001011, 13, 7, 0), (0b0000000001110, 13, 7, 1),
    (0b00000000101, 11, 7, 2), (0b00001001, 8, 7, 3), (0b0000000001000, 13, 8, 0), (0b0000000001101, 13, 8, 1),
    (0b0000000001010, 13, 8, 2), (0b000000000100, 12, 8, 3), (0b00000000001111, 14, 9, 0), (0b00000000001110, 14, 9, 1),
    (0b0000000001001, 13, 9, 2), (0b000000000101, 12, 9, 3), (0b000000000001111, 15, 10, 0), (0b000000000001110, 15, 10, 1),
    (0b00000000001101, 14, 10, 2), (0b000000000110, 12, 10, 3), (0b0000000000001111, 16, 11, 0), (0b000000000001011, 15, 11, 1),
    (0b00000000001100, 14, 11, 2), (0b000000000111, 12, 11, 3), (0b0000000000001011, 16, 12, 0), (0b0000000000001000, 16, 12, 1),
    (0b000000000001010, 15, 12, 2), (0b00000000001011, 14, 12, 3), (0b0000000000001001, 16, 13, 0), (0b0000000000001110, 16, 13, 1),
    (0b000000000001101, 15, 13, 2), (0b00000000001010, 14, 13, 3), (0b0000000000000111, 16, 14, 0), (0b0000000000001010, 16, 14, 1),
    (0b000000000001100, 15, 14, 2), (0b00000000001001, 14, 14, 3), (0b0000000000000100, 16, 15, 0), (0b0000000000000110, 16, 15, 1),
    (0b0000000000001101, 16, 15, 2), (0b00000000001000, 14, 15, 3), (0b0000000000000101, 16, 16, 0), (0b0000000000000001, 16, 16, 1),
    (0b0000000000000010, 16, 16, 2), (0b0000000000001100, 16, 16, 3),
];

fn try_match(bits: &[u8]) -> Option<(u8, u8, usize)> {
    let mut code = 0u32;
    for len in 1..=16.min(bits.len()) {
        code = (code << 1) | (bits[len - 1] as u32);
        for &(cw, cwlen, tc, to) in &COEFF_TOKEN_NC0 {
            if cwlen as usize == len && cw == code {
                return Some((tc, to, len));
            }
        }
    }
    None
}

fn main() {
    let data = std::fs::read("testdata/i4x4_frame.h264").unwrap();
    let nals = parse_annex_b(&data);
    let idr = nals
        .iter()
        .find(|n| n.nal_unit_type == NalUnitType::SliceIdr)
        .unwrap();
    let rbsp = &idr.rbsp;

    println!("RBSP length: {}", rbsp.len());

    // Print bytes 7-12
    println!("Bytes 7-12:");
    for i in 7..=12.min(rbsp.len() - 1) {
        println!("  byte {}: 0x{:02x} = {:08b}", i, rbsp[i], rbsp[i]);
    }

    println!("\nTrying coeff_token match at different bit positions (around bit 71):");

    for start_bit in 68..76 {
        let byte = start_bit / 8;
        let bit_in_byte = start_bit % 8;

        if byte >= rbsp.len() {
            continue;
        }

        // Extract next 20 bits from this position
        let mut bits = Vec::new();
        let mut cur_byte = byte;
        let mut cur_bit = bit_in_byte;

        for _ in 0..20 {
            if cur_byte >= rbsp.len() {
                break;
            }
            let b = (rbsp[cur_byte] >> (7 - cur_bit)) & 1;
            bits.push(b);
            cur_bit += 1;
            if cur_bit == 8 {
                cur_bit = 0;
                cur_byte += 1;
            }
        }

        let bit_str: String = bits
            .iter()
            .take(16)
            .map(|&b| if b == 1 { '1' } else { '0' })
            .collect();

        if let Some((tc, to, len)) = try_match(&bits) {
            println!(
                "bit {:2} = ({}, {}): [{}] -> tc={}, to={}, len={}",
                start_bit, byte, bit_in_byte, bit_str, tc, to, len
            );
        } else {
            println!(
                "bit {:2} = ({}, {}): [{}] -> no 16-bit match",
                start_bit, byte, bit_in_byte, bit_str
            );
        }
    }

    // Also try if we're 1 byte off
    println!("\nTrying positions if we're 1 byte ahead (around bit 79):");
    for start_bit in 76..84 {
        let byte = start_bit / 8;
        let bit_in_byte = start_bit % 8;

        if byte >= rbsp.len() {
            continue;
        }

        let mut bits = Vec::new();
        let mut cur_byte = byte;
        let mut cur_bit = bit_in_byte;

        for _ in 0..20 {
            if cur_byte >= rbsp.len() {
                break;
            }
            let b = (rbsp[cur_byte] >> (7 - cur_bit)) & 1;
            bits.push(b);
            cur_bit += 1;
            if cur_bit == 8 {
                cur_bit = 0;
                cur_byte += 1;
            }
        }

        let bit_str: String = bits
            .iter()
            .take(16)
            .map(|&b| if b == 1 { '1' } else { '0' })
            .collect();

        if let Some((tc, to, len)) = try_match(&bits) {
            println!(
                "bit {:2} = ({}, {}): [{}] -> tc={}, to={}, len={}",
                start_bit, byte, bit_in_byte, bit_str, tc, to, len
            );
        }
    }
}
