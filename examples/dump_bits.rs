use rust_h264::nal::{parse_annex_b, NalUnitType};

fn main() {
    let data = std::fs::read("/tmp/bars.h264").unwrap();
    let nals = parse_annex_b(&data);
    let idr = nals.iter().find(|n| n.nal_unit_type == NalUnitType::SliceIdr).unwrap();
    
    println!("IDR RBSP bytes 0-20:");
    for i in 0..20.min(idr.rbsp.len()) {
        print!("{:02x} ", idr.rbsp[i]);
        if (i + 1) % 10 == 0 { println!(); }
    }
    println!();
    
    // Position (3, 7) is bit 0 of byte 3
    println!("\nBits starting from (3, 7):");
    let mut bits = Vec::new();
    for byte_idx in 3..13 {
        if byte_idx < idr.rbsp.len() {
            for bit_idx in 0..8 {
                let b = (idr.rbsp[byte_idx] >> (7 - bit_idx)) & 1;
                bits.push(b);
            }
        }
    }
    
    // Print bits grouped by byte
    for (i, b) in bits.iter().enumerate() {
        print!("{}", b);
        if (i + 1) % 8 == 0 { print!(" "); }
    }
    println!();
    
    // Starting from bit 7 of byte 3 (after slice header + mb_type)
    println!("\nBits from position (3, 7) specifically:");
    let start_bit = 7;  // bit 0 (LSB) of byte 3
    let mut idx = 0;
    for byte_idx in 3..13 {
        let byte = idr.rbsp[byte_idx];
        for bit_idx in (if byte_idx == 3 { start_bit } else { 0 })..8 {
            let b = (byte >> (7 - bit_idx)) & 1;
            print!("{}", b);
            idx += 1;
            if idx % 8 == 0 { print!(" "); }
        }
    }
    println!();
    
    // Parse coeff_token manually
    println!("\nManual coeff_token parse (nc=0):");
    // Starting from (3, 7), first bit is byte 3 bit 0 = LSB
    let byte3 = idr.rbsp[3];
    println!("Byte 3: 0x{:02x} = {:08b}", byte3, byte3);
    println!("Bit 7 of byte 3 (our start): {}", (byte3 >> 0) & 1);
}
