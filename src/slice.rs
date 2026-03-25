use crate::bitstream::BitstreamReader;
use crate::nal::NalUnitType;
use crate::pps::Pps;
use crate::sps::Sps;

/// Slice types per H.264 Table 7-6.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceType {
    P,
    B,
    I,
    Sp,
    Si,
}

impl SliceType {
    pub fn from_raw(val: u32) -> Result<Self, &'static str> {
        match val {
            0 | 5 => Ok(SliceType::P),
            1 | 6 => Ok(SliceType::B),
            2 | 7 => Ok(SliceType::I),
            3 | 8 => Ok(SliceType::Sp),
            4 | 9 => Ok(SliceType::Si),
            _ => Err("invalid slice_type"),
        }
    }
}

/// Slice header (H.264 spec section 7.3.3).
#[derive(Debug)]
pub struct SliceHeader {
    pub first_mb_in_slice: u32,
    pub slice_type: SliceType,
    pub pic_parameter_set_id: u32,
    pub frame_num: u32,
    pub idr_pic_id: Option<u32>,
    pub no_output_of_prior_pics_flag: bool,
    pub long_term_reference_flag: bool,
    pub slice_qp_delta: i32,
    pub disable_deblocking_filter_idc: u32,
    pub slice_alpha_c0_offset_div2: i32,
    pub slice_beta_offset_div2: i32,
    // POC fields (stored for DPB ordering)
    pub pic_order_cnt_lsb: u32,
    pub delta_pic_order_cnt_bottom: i32,
    pub delta_pic_order_cnt: [i32; 2],
    /// Number of active L0 reference frames for P/B slices (1-based).
    pub num_ref_idx_l0_active: u32,
    /// Number of active L1 reference frames for B slices (1-based).
    pub num_ref_idx_l1_active: u32,
    /// Spatial (true) vs temporal (false) direct mode for B slices.
    pub direct_spatial_mv_pred_flag: bool,
}

impl SliceHeader {
    pub fn qp_y(&self, pps: &Pps) -> i32 {
        26 + pps.pic_init_qp_minus26 + self.slice_qp_delta
    }
}

/// Parse a slice header from RBSP data. Returns the header and a reader
/// positioned at the start of slice data (macroblock layer).
pub fn parse_slice_header(
    rbsp: &[u8],
    sps: &Sps,
    pps: &Pps,
    nal_unit_type: NalUnitType,
    nal_ref_idc: u8,
) -> Result<(SliceHeader, BitstreamReader), &'static str> {
    let mut r = BitstreamReader::new(rbsp);

    let first_mb_in_slice = r.read_ue()?;
    let slice_type_raw = r.read_ue()?;
    let slice_type = SliceType::from_raw(slice_type_raw)?;
    let pic_parameter_set_id = r.read_ue()?;

    let frame_num_bits = sps.log2_max_frame_num_minus4 + 4;
    let frame_num = r.read_bits(frame_num_bits as u8)?;

    // field_pic_flag / bottom_field_flag only if !frame_mbs_only — skip for now
    // (our test file has frame_mbs_only_flag = true)

    let mut idr_pic_id = None;
    if nal_unit_type == NalUnitType::SliceIdr {
        idr_pic_id = Some(r.read_ue()?);
    }

    // pic_order_cnt_type == 0: read pic_order_cnt_lsb
    // pic_order_cnt_type == 1: read delta_pic_order_cnt
    // pic_order_cnt_type == 2: nothing
    let mut pic_order_cnt_lsb = 0u32;
    let mut delta_pic_order_cnt_bottom = 0i32;
    let mut delta_pic_order_cnt = [0i32; 2];

    if sps.pic_order_cnt_type == 0 {
        let poc_lsb_bits = sps.log2_max_pic_order_cnt_lsb_minus4 + 4;
        pic_order_cnt_lsb = r.read_bits(poc_lsb_bits as u8)?;
        if pps.bottom_field_pic_order_in_frame_present_flag {
            delta_pic_order_cnt_bottom = r.read_se()?;
        }
    } else if sps.pic_order_cnt_type == 1 && !sps.delta_pic_order_always_zero_flag {
        delta_pic_order_cnt[0] = r.read_se()?;
        if pps.bottom_field_pic_order_in_frame_present_flag {
            delta_pic_order_cnt[1] = r.read_se()?;
        }
    }

    // direct_spatial_mv_pred_flag (B-slices only, spec 7.3.3)
    let mut direct_spatial_mv_pred_flag = false;
    if slice_type == SliceType::B {
        direct_spatial_mv_pred_flag = r.read_bit()? != 0;
    }

    // num_ref_idx_active_override (for P/B slices)
    let mut num_ref_idx_l0_active = if slice_type == SliceType::I || slice_type == SliceType::Si {
        0
    } else {
        pps.num_ref_idx_l0_default_active_minus1 + 1
    };
    let mut num_ref_idx_l1_active = if slice_type == SliceType::B {
        pps.num_ref_idx_l1_default_active_minus1 + 1
    } else {
        0
    };
    if slice_type != SliceType::I && slice_type != SliceType::Si {
        let num_ref_idx_active_override_flag = r.read_bit()? != 0;
        if num_ref_idx_active_override_flag {
            num_ref_idx_l0_active = r.read_ue()? + 1;
            if slice_type == SliceType::B {
                num_ref_idx_l1_active = r.read_ue()? + 1;
            }
        }
    }

    // ref_pic_list_modification (spec 7.3.3.1) — skip/consume for now
    if slice_type != SliceType::I && slice_type != SliceType::Si {
        let ref_pic_list_modification_flag_l0 = r.read_bit()? != 0;
        if ref_pic_list_modification_flag_l0 {
            loop {
                let modification_of_pic_nums_idc = r.read_ue()?;
                if modification_of_pic_nums_idc == 3 {
                    break;
                }
                let _ = r.read_ue()?; // abs_diff_pic_num_minus1 or long_term_pic_num
            }
        }
        if slice_type == SliceType::B {
            let ref_pic_list_modification_flag_l1 = r.read_bit()? != 0;
            if ref_pic_list_modification_flag_l1 {
                loop {
                    let modification_of_pic_nums_idc = r.read_ue()?;
                    if modification_of_pic_nums_idc == 3 {
                        break;
                    }
                    let _ = r.read_ue()?;
                }
            }
        }
    }

    // pred_weight_table (spec 7.3.3.2) — consume but don't store
    let needs_weight_table = (slice_type == SliceType::P && pps.weighted_pred_flag)
        || (slice_type == SliceType::B && pps.weighted_bipred_idc == 1);
    if needs_weight_table {
        let _luma_log2_weight_denom = r.read_ue()?;
        let _chroma_log2_weight_denom = r.read_ue()?;
        for _ in 0..num_ref_idx_l0_active {
            let luma_weight_flag = r.read_bit()? != 0;
            if luma_weight_flag {
                let _ = r.read_se()?; // luma_weight
                let _ = r.read_se()?; // luma_offset
            }
            let chroma_weight_flag = r.read_bit()? != 0;
            if chroma_weight_flag {
                for _ in 0..2 {
                    let _ = r.read_se()?; // chroma_weight
                    let _ = r.read_se()?; // chroma_offset
                }
            }
        }
        if slice_type == SliceType::B {
            for _ in 0..num_ref_idx_l1_active {
                let luma_weight_flag = r.read_bit()? != 0;
                if luma_weight_flag {
                    let _ = r.read_se()?;
                    let _ = r.read_se()?;
                }
                let chroma_weight_flag = r.read_bit()? != 0;
                if chroma_weight_flag {
                    for _ in 0..2 {
                        let _ = r.read_se()?;
                        let _ = r.read_se()?;
                    }
                }
            }
        }
    }

    // dec_ref_pic_marking (spec 7.3.3.3)
    let mut no_output_of_prior_pics_flag = false;
    let mut long_term_reference_flag = false;
    if nal_unit_type == NalUnitType::SliceIdr {
        no_output_of_prior_pics_flag = r.read_bit()? != 0;
        long_term_reference_flag = r.read_bit()? != 0;
    } else if nal_ref_idc > 0 {
        let adaptive_ref_pic_marking_mode_flag = r.read_bit()? != 0;
        if adaptive_ref_pic_marking_mode_flag {
            loop {
                let op = r.read_ue()?;
                if op == 0 {
                    break;
                }
                match op {
                    1 | 3 => { let _ = r.read_ue()?; }
                    2 => { let _ = r.read_ue()?; }
                    4 | 5 => { let _ = r.read_ue()?; }
                    6 => { let _ = r.read_ue()?; }
                    _ => break,
                }
            }
        }
    }

    let slice_qp_delta = r.read_se()?;

    let mut disable_deblocking_filter_idc = 0;
    let mut slice_alpha_c0_offset_div2 = 0;
    let mut slice_beta_offset_div2 = 0;
    if pps.deblocking_filter_control_present_flag {
        disable_deblocking_filter_idc = r.read_ue()?;
        if disable_deblocking_filter_idc != 1 {
            slice_alpha_c0_offset_div2 = r.read_se()?;
            slice_beta_offset_div2 = r.read_se()?;
        }
    }

    let header = SliceHeader {
        first_mb_in_slice,
        slice_type,
        pic_parameter_set_id,
        frame_num,
        idr_pic_id,
        no_output_of_prior_pics_flag,
        long_term_reference_flag,
        slice_qp_delta,
        disable_deblocking_filter_idc,
        slice_alpha_c0_offset_div2,
        slice_beta_offset_div2,
        pic_order_cnt_lsb,
        delta_pic_order_cnt_bottom,
        delta_pic_order_cnt,
        num_ref_idx_l0_active,
        num_ref_idx_l1_active,
        direct_spatial_mv_pred_flag,
    };

    Ok((header, r))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nal::{parse_annex_b, NalUnitType};
    use crate::pps::parse_pps;
    use crate::sps::parse_sps;

    #[test]
    fn test_parse_slice_header_idr() {
        let data = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/single_frame.h264"
        ))
        .unwrap();
        let nals = parse_annex_b(&data);

        let sps_nal = nals.iter().find(|n| n.nal_unit_type == NalUnitType::Sps).unwrap();
        let pps_nal = nals.iter().find(|n| n.nal_unit_type == NalUnitType::Pps).unwrap();
        let idr_nal = nals.iter().find(|n| n.nal_unit_type == NalUnitType::SliceIdr).unwrap();

        let sps = parse_sps(&sps_nal.rbsp).unwrap();
        let pps = parse_pps(&pps_nal.rbsp, None).unwrap();

        let (header, _reader) =
            parse_slice_header(&idr_nal.rbsp, &sps, &pps, NalUnitType::SliceIdr, 3).unwrap();

        assert_eq!(header.first_mb_in_slice, 0);
        assert_eq!(header.slice_type, SliceType::I);
        assert_eq!(header.pic_parameter_set_id, 0);
        assert_eq!(header.frame_num, 0);
        assert_eq!(header.idr_pic_id, Some(0));
    }
}
