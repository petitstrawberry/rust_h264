use crate::bitstream::BitstreamReader;

/// H.264 Table 7-2: Default 4x4 scaling list for Intra (in scan order).
#[rustfmt::skip]
pub const DEFAULT_SCALING_4X4_INTRA: [u8; 16] = [
     6, 13, 13, 20,
    20, 20, 28, 28,
    28, 28, 32, 32,
    32, 37, 37, 42,
];

/// H.264 Table 7-2: Default 4x4 scaling list for Inter (in scan order).
#[rustfmt::skip]
pub const DEFAULT_SCALING_4X4_INTER: [u8; 16] = [
    10, 14, 14, 20,
    20, 20, 24, 24,
    24, 24, 27, 27,
    27, 30, 30, 34,
];

/// Flat scaling list (no custom scaling) — all 16s.
pub const FLAT_SCALING_4X4: [u8; 16] = [16; 16];

/// Sequence Parameter Set (H.264 spec section 7.3.2.1).
#[derive(Debug)]
pub struct Sps {
    pub profile_idc: u8,
    pub constraint_set0_flag: bool,
    pub constraint_set1_flag: bool,
    pub constraint_set2_flag: bool,
    pub constraint_set3_flag: bool,
    pub constraint_set4_flag: bool,
    pub constraint_set5_flag: bool,
    pub level_idc: u8,
    pub seq_parameter_set_id: u32,

    // High profile and above fields
    pub chroma_format_idc: u32,
    pub separate_colour_plane_flag: bool,
    pub bit_depth_luma_minus8: u32,
    pub bit_depth_chroma_minus8: u32,
    pub qpprime_y_zero_transform_bypass_flag: bool,
    pub seq_scaling_matrix_present_flag: bool,
    /// 4x4 scaling matrices [0..5]: Intra Y, Intra Cb, Intra Cr, Inter Y, Inter Cb, Inter Cr.
    /// Default is all 16s (flat scaling). Stored in raster scan order within each 4x4 block.
    pub scaling_list_4x4: [[u8; 16]; 6],
    /// 8x8 scaling matrices [0..1]: Intra Y, Inter Y (for 4:2:0).
    /// Default is all 16s. Stored in raster scan order within each 8x8 block.
    pub scaling_list_8x8: [[u8; 64]; 2],

    pub log2_max_frame_num_minus4: u32,
    pub pic_order_cnt_type: u32,
    pub log2_max_pic_order_cnt_lsb_minus4: u32,
    pub delta_pic_order_always_zero_flag: bool,
    pub offset_for_non_ref_pic: i32,
    pub offset_for_top_to_bottom_field: i32,
    pub num_ref_frames_in_pic_order_cnt_cycle: u32,
    pub offset_for_ref_frame: Vec<i32>,

    pub max_num_ref_frames: u32,
    pub gaps_in_frame_num_value_allowed_flag: bool,
    pub pic_width_in_mbs_minus1: u32,
    pub pic_height_in_map_units_minus1: u32,
    pub frame_mbs_only_flag: bool,
    pub mb_adaptive_frame_field_flag: bool,
    pub direct_8x8_inference_flag: bool,

    pub frame_cropping_flag: bool,
    pub frame_crop_left_offset: u32,
    pub frame_crop_right_offset: u32,
    pub frame_crop_top_offset: u32,
    pub frame_crop_bottom_offset: u32,

    pub vui_parameters_present_flag: bool,
}

impl Sps {
    /// Width in pixels (accounting for cropping).
    pub fn width(&self) -> u32 {
        let crop_x = if self.frame_cropping_flag {
            (self.frame_crop_left_offset + self.frame_crop_right_offset) * self.crop_unit_x()
        } else {
            0
        };
        (self.pic_width_in_mbs_minus1 + 1) * 16 - crop_x
    }

    /// Height in pixels (accounting for cropping).
    pub fn height(&self) -> u32 {
        let crop_y = if self.frame_cropping_flag {
            (self.frame_crop_top_offset + self.frame_crop_bottom_offset) * self.crop_unit_y()
        } else {
            0
        };
        (self.pic_height_in_map_units_minus1 + 1)
            * 16
            * (if self.frame_mbs_only_flag { 1 } else { 2 })
            - crop_y
    }

    fn crop_unit_x(&self) -> u32 {
        if self.chroma_format_idc == 0 {
            1
        } else {
            // SubWidthC: 2 for 4:2:0 and 4:2:2, 1 for 4:4:4
            if self.chroma_format_idc == 3 {
                1
            } else {
                2
            }
        }
    }

    fn crop_unit_y(&self) -> u32 {
        let sub_height_c = if self.chroma_format_idc == 1 { 2 } else { 1 };
        let frame_factor = if self.frame_mbs_only_flag { 1 } else { 2 };
        if self.chroma_format_idc == 0 {
            frame_factor
        } else {
            sub_height_c * frame_factor
        }
    }
}

fn is_high_profile(profile_idc: u8) -> bool {
    matches!(
        profile_idc,
        100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128 | 138 | 139 | 134 | 135
    )
}

/// Parse an SPS from RBSP data (NAL header byte already stripped).
pub fn parse_sps(rbsp: &[u8]) -> Result<Sps, &'static str> {
    let mut r = BitstreamReader::new(rbsp);

    let profile_idc = r.read_bits(8)? as u8;
    let constraint_set0_flag = r.read_bit()? != 0;
    let constraint_set1_flag = r.read_bit()? != 0;
    let constraint_set2_flag = r.read_bit()? != 0;
    let constraint_set3_flag = r.read_bit()? != 0;
    let constraint_set4_flag = r.read_bit()? != 0;
    let constraint_set5_flag = r.read_bit()? != 0;
    let _reserved_zero_2bits = r.read_bits(2)?;
    let level_idc = r.read_bits(8)? as u8;
    let seq_parameter_set_id = r.read_ue()?;

    let mut chroma_format_idc = 1; // default
    let mut separate_colour_plane_flag = false;
    let mut bit_depth_luma_minus8 = 0;
    let mut bit_depth_chroma_minus8 = 0;
    let mut qpprime_y_zero_transform_bypass_flag = false;
    let mut seq_scaling_matrix_present_flag = false;
    // Default: flat scaling (all 16s) when seq_scaling_matrix_present_flag is false
    let mut scaling_list_4x4 = [FLAT_SCALING_4X4; 6];
    let mut scaling_list_8x8 = [[16u8; 64]; 2];

    if is_high_profile(profile_idc) {
        chroma_format_idc = r.read_ue()?;
        if chroma_format_idc == 3 {
            separate_colour_plane_flag = r.read_bit()? != 0;
        }
        bit_depth_luma_minus8 = r.read_ue()?;
        bit_depth_chroma_minus8 = r.read_ue()?;
        qpprime_y_zero_transform_bypass_flag = r.read_bit()? != 0;
        seq_scaling_matrix_present_flag = r.read_bit()? != 0;
        if seq_scaling_matrix_present_flag {
            let count = if chroma_format_idc != 3 { 8 } else { 12 };
            for i in 0..count {
                let present = r.read_bit()? != 0;
                if present {
                    if i < 6 {
                        scaling_list_4x4[i] = parse_scaling_list::<16>(&mut r, 16)?;
                    } else if i < 8 {
                        scaling_list_8x8[i - 6] = parse_scaling_list::<64>(&mut r, 64)?;
                    } else {
                        let _: [u8; 64] = parse_scaling_list::<64>(&mut r, 64)?;
                    }
                } else if i < 6 {
                    // Fallback per H.264 Table 7-2:
                    // i=0: Default_4x4_Intra, i=3: Default_4x4_Inter
                    // i=1,2: copy from previous, i=4,5: copy from previous
                    scaling_list_4x4[i] = match i {
                        0 => DEFAULT_SCALING_4X4_INTRA,
                        3 => DEFAULT_SCALING_4X4_INTER,
                        _ => scaling_list_4x4[i - 1],
                    };
                }
            }
        }
    }

    let log2_max_frame_num_minus4 = r.read_ue()?;
    let pic_order_cnt_type = r.read_ue()?;

    let mut log2_max_pic_order_cnt_lsb_minus4 = 0;
    let mut delta_pic_order_always_zero_flag = false;
    let mut offset_for_non_ref_pic = 0;
    let mut offset_for_top_to_bottom_field = 0;
    let mut num_ref_frames_in_pic_order_cnt_cycle = 0;
    let mut offset_for_ref_frame = Vec::new();

    if pic_order_cnt_type == 0 {
        log2_max_pic_order_cnt_lsb_minus4 = r.read_ue()?;
    } else if pic_order_cnt_type == 1 {
        delta_pic_order_always_zero_flag = r.read_bit()? != 0;
        offset_for_non_ref_pic = r.read_se()?;
        offset_for_top_to_bottom_field = r.read_se()?;
        num_ref_frames_in_pic_order_cnt_cycle = r.read_ue()?;
        for _ in 0..num_ref_frames_in_pic_order_cnt_cycle {
            offset_for_ref_frame.push(r.read_se()?);
        }
    }

    let max_num_ref_frames = r.read_ue()?;
    let gaps_in_frame_num_value_allowed_flag = r.read_bit()? != 0;
    let pic_width_in_mbs_minus1 = r.read_ue()?;
    let pic_height_in_map_units_minus1 = r.read_ue()?;
    let frame_mbs_only_flag = r.read_bit()? != 0;

    let mut mb_adaptive_frame_field_flag = false;
    if !frame_mbs_only_flag {
        mb_adaptive_frame_field_flag = r.read_bit()? != 0;
    }

    let direct_8x8_inference_flag = r.read_bit()? != 0;

    let frame_cropping_flag = r.read_bit()? != 0;
    let mut frame_crop_left_offset = 0;
    let mut frame_crop_right_offset = 0;
    let mut frame_crop_top_offset = 0;
    let mut frame_crop_bottom_offset = 0;
    if frame_cropping_flag {
        frame_crop_left_offset = r.read_ue()?;
        frame_crop_right_offset = r.read_ue()?;
        frame_crop_top_offset = r.read_ue()?;
        frame_crop_bottom_offset = r.read_ue()?;
    }

    let vui_parameters_present_flag = r.read_bit()? != 0;
    // VUI parsing skipped for now

    Ok(Sps {
        profile_idc,
        constraint_set0_flag,
        constraint_set1_flag,
        constraint_set2_flag,
        constraint_set3_flag,
        constraint_set4_flag,
        constraint_set5_flag,
        level_idc,
        seq_parameter_set_id,
        chroma_format_idc,
        separate_colour_plane_flag,
        bit_depth_luma_minus8,
        bit_depth_chroma_minus8,
        qpprime_y_zero_transform_bypass_flag,
        seq_scaling_matrix_present_flag,
        scaling_list_4x4,
        scaling_list_8x8,
        log2_max_frame_num_minus4,
        pic_order_cnt_type,
        log2_max_pic_order_cnt_lsb_minus4,
        delta_pic_order_always_zero_flag,
        offset_for_non_ref_pic,
        offset_for_top_to_bottom_field,
        num_ref_frames_in_pic_order_cnt_cycle,
        offset_for_ref_frame,
        max_num_ref_frames,
        gaps_in_frame_num_value_allowed_flag,
        pic_width_in_mbs_minus1,
        pic_height_in_map_units_minus1,
        frame_mbs_only_flag,
        mb_adaptive_frame_field_flag,
        direct_8x8_inference_flag,
        frame_cropping_flag,
        frame_crop_left_offset,
        frame_crop_right_offset,
        frame_crop_top_offset,
        frame_crop_bottom_offset,
        vui_parameters_present_flag,
    })
}

/// Parse a scaling list from the bitstream (H.264 spec 7.3.2.1.1).
/// Returns a flat array of `size` scale values in scan order.
pub fn parse_scaling_list<const N: usize>(
    r: &mut BitstreamReader,
    size: usize,
) -> Result<[u8; N], &'static str> {
    let mut scaling_list = [0u8; N];
    let mut last_scale: i32 = 8;
    let mut next_scale: i32 = 8;
    for entry in scaling_list.iter_mut().take(size) {
        if next_scale != 0 {
            let delta = r.read_se()?;
            next_scale = (last_scale + delta + 256) % 256;
        }
        let val = if next_scale == 0 {
            last_scale
        } else {
            next_scale
        };
        *entry = val as u8;
        last_scale = val;
    }
    Ok(scaling_list)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nal::{parse_annex_b, NalUnitType};

    #[test]
    fn test_parse_sps_single_frame() {
        let data = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/single_frame.h264"
        ))
        .unwrap();
        let nals = parse_annex_b(&data);
        let sps_nal = nals
            .iter()
            .find(|n| n.nal_unit_type == NalUnitType::Sps)
            .unwrap();
        let sps = parse_sps(&sps_nal.rbsp).unwrap();

        assert_eq!(sps.profile_idc, 66); // Baseline
        assert!(sps.constraint_set1_flag); // Constrained Baseline
        assert_eq!(sps.level_idc, 10); // Level 1.0
        assert_eq!(sps.seq_parameter_set_id, 0);
        assert_eq!(sps.chroma_format_idc, 1); // 4:2:0 (default for baseline)
        assert_eq!(sps.width(), 16);
        assert_eq!(sps.height(), 16);
        assert!(sps.frame_mbs_only_flag);
        assert_eq!(sps.max_num_ref_frames, 0);
        assert_eq!(sps.pic_order_cnt_type, 2);
    }
}
