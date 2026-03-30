use std::borrow::Cow;

/// NAL unit types relevant to SPS/PPS parsing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NalUnitType {
    Slice,               // 1
    SliceDataA,          // 2
    SliceDataB,          // 3
    SliceDataC,          // 4
    SliceIdr,            // 5
    Sei,                 // 6
    Sps,                 // 7
    Pps,                 // 8
    AccessUnitDelimiter, // 9
    EndOfSequence,       // 10
    EndOfStream,         // 11
    FillerData,          // 12
    Other(u8),
}

impl From<u8> for NalUnitType {
    fn from(val: u8) -> Self {
        match val {
            1 => NalUnitType::Slice,
            2 => NalUnitType::SliceDataA,
            3 => NalUnitType::SliceDataB,
            4 => NalUnitType::SliceDataC,
            5 => NalUnitType::SliceIdr,
            6 => NalUnitType::Sei,
            7 => NalUnitType::Sps,
            8 => NalUnitType::Pps,
            9 => NalUnitType::AccessUnitDelimiter,
            10 => NalUnitType::EndOfSequence,
            11 => NalUnitType::EndOfStream,
            12 => NalUnitType::FillerData,
            v => NalUnitType::Other(v),
        }
    }
}

#[derive(Debug)]
pub struct NalUnit<'a> {
    pub nal_ref_idc: u8,
    pub nal_unit_type: NalUnitType,
    /// RBSP data (emulation prevention bytes removed, or borrowed directly
    /// from the input when no emulation prevention bytes are present).
    pub rbsp: Cow<'a, [u8]>,
}

/// Split an Annex B bytestream into NAL units.
/// Handles both 3-byte (00 00 01) and 4-byte (00 00 00 01) start codes.
pub fn parse_annex_b(data: &[u8]) -> Vec<NalUnit<'_>> {
    let mut nals = Vec::new();
    // Find first start code
    let mut i = match find_start_code(data, 0) {
        Some((pos, _)) => pos,
        None => return nals,
    };

    loop {
        // i points to first byte after start code (the NAL header byte)
        if i >= data.len() {
            break;
        }

        // Find the next start code to determine where this NAL ends
        let nal_end = match find_start_code(data, i) {
            Some((pos, sc_start)) => {
                let end = sc_start;
                // Strip trailing zeros before the start code
                let mut e = end;
                while e > i && data[e - 1] == 0 {
                    e -= 1;
                }
                (e, Some(pos))
            }
            None => (data.len(), None),
        };

        let nal_data = &data[i..nal_end.0];
        if !nal_data.is_empty() {
            let header = nal_data[0];
            // forbidden_zero_bit (MSB) must be 0; skip invalid NAL units
            if header & 0x80 == 0 {
                let nal_ref_idc = (header >> 5) & 0x03;
                let nal_unit_type = NalUnitType::from(header & 0x1F);
                let rbsp = remove_emulation_prevention(&nal_data[1..]);
                nals.push(NalUnit {
                    nal_ref_idc,
                    nal_unit_type,
                    rbsp,
                });
            }
        }

        match nal_end.1 {
            Some(pos) => i = pos,
            None => break,
        }
    }

    nals
}

/// Find the next start code starting from `offset`.
/// Returns (position after start code, position of start code beginning).
fn find_start_code(data: &[u8], offset: usize) -> Option<(usize, usize)> {
    let mut i = offset;
    while i + 2 < data.len() {
        if data[i] == 0 && data[i + 1] == 0 {
            if data[i + 2] == 1 {
                return Some((i + 3, i));
            }
            if i + 3 < data.len() && data[i + 2] == 0 && data[i + 3] == 1 {
                return Some((i + 4, i));
            }
        }
        i += 1;
    }
    None
}

/// Remove emulation prevention bytes (0x03 in 00 00 03 sequences).
/// Returns a borrowed slice when no emulation prevention bytes are found
/// (the common case), avoiding allocation entirely.
fn remove_emulation_prevention(data: &[u8]) -> Cow<'_, [u8]> {
    // Fast path: scan for 00 00 03. If none found, return borrowed slice.
    let has_epb = data.windows(3).any(|w| w[0] == 0 && w[1] == 0 && w[2] == 3);
    if !has_epb {
        return Cow::Borrowed(data);
    }

    // Slow path: copy with emulation prevention removal.
    let mut rbsp = Vec::with_capacity(data.len());
    let mut i = 0;
    while i < data.len() {
        if i + 2 < data.len() && data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 3 {
            rbsp.push(0);
            rbsp.push(0);
            i += 3; // skip the 0x03 byte
        } else {
            rbsp.push(data[i]);
            i += 1;
        }
    }
    Cow::Owned(rbsp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_annex_b_single_frame() {
        let data = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/single_frame.h264"
        ))
        .unwrap();
        let nals = parse_annex_b(&data);
        assert_eq!(nals.len(), 4);
        assert_eq!(nals[0].nal_unit_type, NalUnitType::Sps);
        assert_eq!(nals[1].nal_unit_type, NalUnitType::Pps);
        assert_eq!(nals[2].nal_unit_type, NalUnitType::Sei);
        assert_eq!(nals[3].nal_unit_type, NalUnitType::SliceIdr);
    }

    #[test]
    fn test_emulation_prevention_removal() {
        let input = [0x00, 0x00, 0x03, 0x01, 0xAB];
        let rbsp = remove_emulation_prevention(&input);
        assert_eq!(&*rbsp, &[0x00, 0x00, 0x01, 0xAB]);
        assert!(
            matches!(rbsp, Cow::Owned(_)),
            "should allocate when EPB present"
        );
    }

    #[test]
    fn test_emulation_prevention_zero_copy() {
        // No emulation prevention bytes → should return borrowed slice (no allocation)
        let input = [0x01, 0x02, 0x03, 0x04];
        let rbsp = remove_emulation_prevention(&input);
        assert_eq!(&*rbsp, &input);
        assert!(
            matches!(rbsp, Cow::Borrowed(_)),
            "should borrow when no EPB"
        );
    }
}
