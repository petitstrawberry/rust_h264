use rust_h264::nal::{parse_annex_b, NalUnitType};
use rust_h264::pps::parse_pps;
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

    println!("SPS:");
    println!("  seq_parameter_set_id: {}", sps.seq_parameter_set_id);
    println!("  profile_idc: {}", sps.profile_idc);
    println!("  level_idc: {}", sps.level_idc);
    println!("  pic_order_cnt_type: {}", sps.pic_order_cnt_type);

    println!("\nPPS:");
    println!("  pic_parameter_set_id: {}", pps.pic_parameter_set_id);
    println!("  seq_parameter_set_id: {}", pps.seq_parameter_set_id);
    println!(
        "  entropy_coding_mode_flag: {}",
        pps.entropy_coding_mode_flag
    );
    println!(
        "  bottom_field_pic_order_in_frame_present_flag: {}",
        pps.bottom_field_pic_order_in_frame_present_flag
    );
    println!("  num_slice_groups_minus1: {}", pps.num_slice_groups_minus1);
    println!(
        "  num_ref_idx_l0_default_active_minus1: {}",
        pps.num_ref_idx_l0_default_active_minus1
    );
    println!("  weighted_pred_flag: {}", pps.weighted_pred_flag);
    println!("  weighted_bipred_idc: {}", pps.weighted_bipred_idc);
    println!("  pic_init_qp_minus26: {}", pps.pic_init_qp_minus26);
    println!("  pic_init_qs_minus26: {}", pps.pic_init_qs_minus26);
    println!("  chroma_qp_index_offset: {}", pps.chroma_qp_index_offset);
    println!(
        "  deblocking_filter_control_present_flag: {}",
        pps.deblocking_filter_control_present_flag
    );
    println!(
        "  constrained_intra_pred_flag: {}",
        pps.constrained_intra_pred_flag
    );
    println!(
        "  redundant_pic_cnt_present_flag: {}",
        pps.redundant_pic_cnt_present_flag
    );
    println!("  transform_8x8_mode_flag: {}", pps.transform_8x8_mode_flag);

    // Also print the raw PPS bytes
    let pps_nal = nals
        .iter()
        .find(|n| n.nal_unit_type == NalUnitType::Pps)
        .unwrap();
    println!("\nPPS RBSP bytes:");
    for (i, b) in pps_nal.rbsp.iter().enumerate() {
        print!("{:02x} ", b);
        if (i + 1) % 16 == 0 {
            println!();
        }
    }
    println!();
}
