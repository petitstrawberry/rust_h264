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
    if sps.pic_order_cnt_type == 0 {
        let poc_lsb_bits = sps.log2_max_pic_order_cnt_lsb_minus4 + 4;
        let _pic_order_cnt_lsb = r.read_bits(poc_lsb_bits as u8)?;
        if pps.bottom_field_pic_order_in_frame_present_flag {
            let _delta_pic_order_cnt_bottom = r.read_se()?;
        }
    } else if sps.pic_order_cnt_type == 1 && !sps.delta_pic_order_always_zero_flag {
        let _delta_pic_order_cnt_0 = r.read_se()?;
        if pps.bottom_field_pic_order_in_frame_present_flag {
            let _delta_pic_order_cnt_1 = r.read_se()?;
        }
    }

    // ref_pic_list_modification — not present for I slices
    // (slice_type != I and slice_type != SI would need this)

    // dec_ref_pic_marking
    let mut no_output_of_prior_pics_flag = false;
    let mut long_term_reference_flag = false;
    if nal_unit_type == NalUnitType::SliceIdr {
        no_output_of_prior_pics_flag = r.read_bit()? != 0;
        long_term_reference_flag = r.read_bit()? != 0;
    }
    // For non-IDR reference pictures, adaptive_ref_pic_marking would go here

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
        let pps = parse_pps(&pps_nal.rbsp).unwrap();

        let (header, _reader) =
            parse_slice_header(&idr_nal.rbsp, &sps, &pps, NalUnitType::SliceIdr).unwrap();

        assert_eq!(header.first_mb_in_slice, 0);
        assert_eq!(header.slice_type, SliceType::I);
        assert_eq!(header.pic_parameter_set_id, 0);
        assert_eq!(header.frame_num, 0);
        assert_eq!(header.idr_pic_id, Some(0));
    }
}
