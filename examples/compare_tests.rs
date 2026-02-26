use rust_h264::nal::{parse_annex_b, NalUnitType};
use rust_h264::sps::parse_sps;
use rust_h264::pps::parse_pps;
use rust_h264::slice::parse_slice_header;

fn analyze_file(name: &str) {
    println!("=== {} ===", name);
    let data = std::fs::read(name).unwrap();
    let nals = parse_annex_b(&data);
    
    let sps = nals.iter().filter(|n| n.nal_unit_type == NalUnitType::Sps)
        .map(|n| parse_sps(&n.rbsp).unwrap()).next().unwrap();
    let pps = nals.iter().filter(|n| n.nal_unit_type == NalUnitType::Pps)
        .map(|n| parse_pps(&n.rbsp).unwrap()).next().unwrap();
    
    println!("SPS: {}x{}, poc_type={}", sps.width(), sps.height(), sps.pic_order_cnt_type);
    println!("PPS: entropy_coding_mode={}, deblocking_filter_ctrl={}",
             pps.entropy_coding_mode_flag, pps.deblocking_filter_control_present_flag);
    
    if let Some(idr) = nals.iter().find(|n| n.nal_unit_type == NalUnitType::SliceIdr) {
        let (header, mut reader) = parse_slice_header(&idr.rbsp, &sps, &pps, NalUnitType::SliceIdr).unwrap();
        println!("Slice: qp_delta={}, qp_y={}", header.slice_qp_delta, header.qp_y(&pps));
        
        let mb_type = reader.read_ue().unwrap();
        println!("First MB: mb_type={}", mb_type);
        
        if mb_type == 0 {
            println!("  I4x4 macroblock");
        } else if mb_type >= 1 && mb_type <= 24 {
            let mt = mb_type - 1;
            println!("  I16x16 macroblock: pred_mode={}, cbp_chroma={}, cbp_luma={}",
                     mt % 4, (mt / 4) % 3, if mt >= 12 { 15 } else { 0 });
        }
        
        // Dump first 20 RBSP bytes of IDR
        println!("IDR RBSP first 20 bytes:");
        for (i, b) in idr.rbsp.iter().take(20).enumerate() {
            print!("{:02x} ", b);
            if (i + 1) % 10 == 0 { println!(); }
        }
        println!();
    }
    println!();
}

fn main() {
    analyze_file("testdata/single_frame.h264");
    analyze_file("testdata/i4x4_frame.h264");
}
