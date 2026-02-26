use rust_h264::nal::{parse_annex_b, NalUnitType};

fn main() {
    let data = std::fs::read("testdata/i4x4_frame.h264").unwrap();
    let nals = parse_annex_b(&data);
    
    for (i, nal) in nals.iter().enumerate() {
        println!("NAL {}: {:?}, rbsp len={}", i, nal.nal_unit_type, nal.rbsp.len());
        if nal.nal_unit_type == NalUnitType::SliceIdr {
            println!("RBSP first 40 bytes:");
            for (j, b) in nal.rbsp.iter().take(40).enumerate() {
                print!("{:02x} ", b);
                if (j + 1) % 16 == 0 {
                    println!();
                }
            }
            println!();
            
            // Find IDR in original data (0x65 NAL type)
            let mut idr_start = 0;
            for i in 0..data.len()-4 {
                if data[i..i+4] == [0,0,0,1] && (data[i+4] & 0x1f) == 5 {
                    idr_start = i + 5; // skip start code and NAL header
                    break;
                }
                if i < data.len()-3 && data[i..i+3] == [0,0,1] && (data[i+3] & 0x1f) == 5 {
                    idr_start = i + 4;
                    break;
                }
            }
            
            if idr_start == 0 {
                println!("Could not find IDR NAL");
                continue;
            }
            
            println!("IDR raw data starts at file offset: 0x{:x}", idr_start);
            
            // Find end of IDR (next start code or EOF)
            let mut idr_end = data.len();
            for i in idr_start..data.len()-3 {
                if data[i..i+3] == [0,0,1] || (i < data.len()-4 && data[i..i+4] == [0,0,0,1]) {
                    idr_end = i;
                    break;
                }
            }
            
            let raw_idr = &data[idr_start..idr_end];
            println!("Raw IDR data len={}", raw_idr.len());
            
            // Check for 00 00 03 sequences
            let mut epb_positions = Vec::new();
            for i in 0..raw_idr.len().saturating_sub(2) {
                if raw_idr[i] == 0 && raw_idr[i+1] == 0 && raw_idr[i+2] == 3 {
                    epb_positions.push(i);
                }
            }
            
            println!("Emulation prevention byte sequences found: {}", epb_positions.len());
            for pos in &epb_positions {
                println!("  EPB at raw offset {}: 00 00 03", pos);
            }
            
            println!("\nRaw IDR first 40 bytes:");
            for (j, b) in raw_idr.iter().take(40).enumerate() {
                print!("{:02x} ", b);
                if (j + 1) % 16 == 0 {
                    println!();
                }
            }
            println!();
            
            // Compare RBSP vs raw
            if nal.rbsp.len() != raw_idr.len() - epb_positions.len() {
                println!("\nWARNING: RBSP length mismatch! Expected {}, got {}", 
                         raw_idr.len() - epb_positions.len(), nal.rbsp.len());
            }
        }
    }
}
