use rust_h264::nal::{parse_annex_b, NalUnitType};
use rust_h264::pps::parse_pps;
use rust_h264::slice::parse_slice_header;
use rust_h264::sps::parse_sps;

fn main() {
    let data = std::fs::read("testdata/i4x4_frame.h264").unwrap();
    let nals = parse_annex_b(&data);

    let sps = nals
        .iter()
        .filter(|n| n.nal_unit_type == NalUnitType::Sps)
        .map(|n| parse_sps(&n.rbsp).unwrap())
        .next()
        .unwrap();
    let pps = nals
        .iter()
        .filter(|n| n.nal_unit_type == NalUnitType::Pps)
        .map(|n| parse_pps(&n.rbsp, None).unwrap())
        .next()
        .unwrap();
    let idr = nals
        .iter()
        .find(|n| n.nal_unit_type == NalUnitType::SliceIdr)
        .unwrap();

    println!("IDR RBSP length: {}", idr.rbsp.len());

    // Parse slice header
    let (_header, mut reader) =
        parse_slice_header(&idr.rbsp, &sps, &pps, NalUnitType::SliceIdr, 3).unwrap();
    let pos = reader.position();
    println!("After slice header: pos=({}, {})", pos.0, pos.1);

    // Read mb_type
    let mb_type = reader.read_ue().unwrap();
    let pos = reader.position();
    println!("After mb_type={}: pos=({}, {})", mb_type, pos.0, pos.1);

    // Read 16 prediction modes
    for _blk in 0..16 {
        let prev_flag = reader.read_bit().unwrap();
        if prev_flag == 0 {
            let _rem = reader.read_bits(3).unwrap();
        }
    }
    let pos = reader.position();
    println!("After 16 pred modes: pos=({}, {})", pos.0, pos.1);

    // Read intra_chroma_pred_mode
    let _icpm = reader.read_ue().unwrap();
    let pos = reader.position();
    println!("After intra_chroma_pred_mode: pos=({}, {})", pos.0, pos.1);

    // Read cbp_code
    let cbp_code = reader.read_ue().unwrap();
    let pos = reader.position();
    println!("After cbp_code={}: pos=({}, {})", cbp_code, pos.0, pos.1);

    // Read mb_qp_delta (if cbp != 0)
    if cbp_code != 3 {
        // cbp_intra[0]=47 != 0
        let _qp_delta = reader.read_se().unwrap();
    }
    let pos = reader.position();
    println!("After mb_qp_delta: pos=({}, {})", pos.0, pos.1);

    // Now we're at CAVLC start for block 0
    // Print next 30 bytes
    println!("\nRBSP bytes from position {} onwards:", pos.0);
    for i in 0..30 {
        if pos.0 + i < idr.rbsp.len() {
            print!("{:02x} ", idr.rbsp[pos.0 + i]);
            if (i + 1) % 16 == 0 {
                println!();
            }
        }
    }
    println!();

    // Print bits starting from current position
    println!("\nFirst 100 bits from ({}, {}):", pos.0, pos.1);
    // Clone reader state conceptually - can't actually clone, so read bits
    let mut bits = Vec::new();
    for _ in 0..100 {
        if let Ok(bit) = reader.read_bit() {
            bits.push(bit);
        } else {
            break;
        }
    }
    for (i, bit) in bits.iter().enumerate() {
        print!("{}", bit);
        if (i + 1) % 8 == 0 {
            print!(" ");
        }
    }
    println!();

    // Now manually look up what VLC this matches
    // For nc=0, table 9-5(a)
    println!("\nFirst 16 bits as coeff_token lookup:");
    let code16: u32 = bits
        .iter()
        .take(16)
        .fold(0u32, |acc, &b| (acc << 1) | b as u32);
    println!("16-bit code: 0b{:016b} = {}", code16, code16);

    // Look for matches at different lengths
    for len in 1..=16 {
        let code: u32 = bits
            .iter()
            .take(len)
            .fold(0u32, |acc, &b| (acc << 1) | b as u32);
        // Check NC0 table entries... too complex here, just print
        if len == 16 && code == 0b0000000000000111 {
            println!("Matched: 16 bits, code=7 -> total_coeff=14, trailing_ones=0");
        }
    }
}
