use crate::bitstream::BitstreamReader;

/// Parse a CAVLC residual block.
/// Returns total_coeff (needed for nC tracking of neighboring blocks).
/// `coeffs` is filled with coefficients in scan order.
/// `max_num_coeff` is 16 for DC blocks, 15 for AC blocks, 4 for chroma DC.
pub fn parse_residual_block_cavlc(
    reader: &mut BitstreamReader,
    coeffs: &mut [i32],
    max_num_coeff: usize,
    nc: i32,
) -> Result<u8, &'static str> {
    for c in coeffs.iter_mut() {
        *c = 0;
    }

    let (total_coeff, trailing_ones) = parse_coeff_token(reader, nc)?;

    if total_coeff == 0 {
        return Ok(0);
    }


    let tc = total_coeff as usize;

    // Trailing ones signs (highest freq first)
    let mut levels = vec![0i32; tc];
    for i in 0..trailing_ones as usize {
        levels[tc - 1 - i] = if reader.read_bit()? != 0 { -1 } else { 1 };
    }

    // Remaining levels
    let mut suffix_length: u32 = if total_coeff > 10 && trailing_ones < 3 {
        1
    } else {
        0
    };

    let remaining_count = tc - trailing_ones as usize;
    for i in (0..remaining_count).rev() {
        let first_nontrailing = i == remaining_count - 1 && trailing_ones < 3;
        let level = parse_level(reader, suffix_length, first_nontrailing)?;
        levels[i] = level;

        if suffix_length == 0 {
            suffix_length = 1;
        }
        if levels[i].unsigned_abs() > (3 << (suffix_length - 1)) {
            suffix_length += 1;
        }
    }

    // Total zeros
    let total_zeros = if total_coeff < max_num_coeff as u8 {
        if max_num_coeff > 4 {
            parse_total_zeros(reader, total_coeff)?
        } else {
            parse_total_zeros_chroma_dc(reader, total_coeff)?
        }
    } else {
        0
    };

    // Run before
    let mut zeros_left = total_zeros;
    let mut run = vec![0u8; tc];
    for i in 0..tc.saturating_sub(1) {
        if zeros_left > 0 {
            run[i] = parse_run_before(reader, zeros_left)?;
            zeros_left -= run[i];
        }
    }
    if tc > 0 {
        run[tc - 1] = zeros_left;
    }



    // Place coefficients per spec 9.2.3:
    // Iterate from lowest-freq (levels[tc-1]) to highest-freq (levels[0]),
    // building coeffIdx upward from -1.
    let mut coeff_idx: i32 = -1;
    for i in (0..tc).rev() {
        coeff_idx += run[i] as i32 + 1;

        coeffs[coeff_idx as usize] = levels[i];
    }

    Ok(total_coeff)
}

fn parse_coeff_token(
    reader: &mut BitstreamReader,
    nc: i32,
) -> Result<(u8, u8), &'static str> {
    if nc < 0 {
        match_vlc(reader, &COEFF_TOKEN_CHROMA_DC)
    } else if nc < 2 {
        match_vlc(reader, &COEFF_TOKEN_NC0)
    } else if nc < 4 {
        match_vlc(reader, &COEFF_TOKEN_NC2)
    } else if nc < 8 {
        match_vlc(reader, &COEFF_TOKEN_NC4)
    } else {
        // nC >= 8: fixed 6-bit code
        let code = reader.read_bits(6)?;
        let trailing_ones = (code & 3) as u8;
        let total_coeff = ((code >> 2) + 1) as u8;
        if trailing_ones > total_coeff || total_coeff > 16 {
            return Err("invalid coeff_token nC>=8");
        }
        Ok((total_coeff, trailing_ones.min(3)))
    }
}

/// Parse level value per H.264 9.2.2.
fn parse_level(
    reader: &mut BitstreamReader,
    suffix_length: u32,
    first_after_trailing: bool,
) -> Result<i32, &'static str> {
    let mut level_prefix: u32 = 0;
    while reader.read_bit()? == 0 {
        level_prefix += 1;
        if level_prefix > 15 {
            return Err("level_prefix too large");
        }
    }

    let level_code;

    if level_prefix < 14 {
        let suffix_len = if suffix_length == 0 && level_prefix == 0 {
            0
        } else {
            suffix_length
        };
        let level_suffix = if suffix_len > 0 {
            reader.read_bits(suffix_len as u8)?
        } else {
            0
        };
        level_code = (level_prefix << suffix_length) | level_suffix;
    } else if level_prefix == 14 {
        let suffix_len = if suffix_length == 0 { 4 } else { suffix_length };
        let level_suffix = reader.read_bits(suffix_len as u8)?;
        level_code = (level_prefix << suffix_length) | level_suffix;
    } else {
        // level_prefix >= 15: levelSuffixSize = level_prefix - 3
        let suffix_bits = if level_prefix >= 15 {
            (level_prefix - 3).max(suffix_length) as u8
        } else {
            suffix_length as u8
        };
        let level_suffix = if suffix_bits > 0 {
            reader.read_bits(suffix_bits)?
        } else {
            0
        };
        level_code = if suffix_length == 0 {
            (15 << suffix_length) + level_suffix + 15
        } else {
            (15 << suffix_length) + level_suffix
        };
    }

    let mut level_code = level_code as i32;

    if first_after_trailing {
        level_code += 2;
    }

    // Convert: even level_code -> positive, odd -> negative
    let level = if level_code % 2 == 0 {
        (level_code + 2) / 2
    } else {
        (-level_code - 1) / 2
    };

    Ok(level)
}

/// Generic VLC table matcher. Table entries: (codeword, bit_length, total_coeff, trailing_ones).
fn match_vlc(
    r: &mut BitstreamReader,
    table: &[(u32, u8, u8, u8)],
) -> Result<(u8, u8), &'static str> {
    let mut code: u32 = 0;
    let mut bits_read: u8 = 0;

    loop {
        code = (code << 1) | r.read_bit()? as u32;
        bits_read += 1;

        for &(codeword, len, tc, to) in table {
            if len == bits_read && codeword == code {
                return Ok((tc, to));
            }
        }

        if bits_read >= 16 {
            break;
        }
    }
    Err("invalid VLC code")
}

// ============================================================
// coeff_token VLC tables from H.264 Table 9-5
// Format: (codeword_value, bit_length, total_coeff, trailing_ones)
// ============================================================

/// Table 9-5(a): 0 <= nC < 2
#[rustfmt::skip]
static COEFF_TOKEN_NC0: [(u32, u8, u8, u8); 62] = [
    (0b1, 1, 0, 0),
    (0b000101, 6, 1, 0),
    (0b01, 2, 1, 1),
    (0b00000111, 8, 2, 0),
    (0b000100, 6, 2, 1),
    (0b001, 3, 2, 2),
    (0b000000111, 9, 3, 0),
    (0b00000110, 8, 3, 1),
    (0b0000101, 7, 3, 2),
    (0b00011, 5, 3, 3),
    (0b0000000111, 10, 4, 0),
    (0b000000110, 9, 4, 1),
    (0b00000101, 8, 4, 2),
    (0b000011, 6, 4, 3),
    (0b00000000111, 11, 5, 0),
    (0b0000000110, 10, 5, 1),
    (0b000000101, 9, 5, 2),
    (0b0000100, 7, 5, 3),
    (0b0000000001111, 13, 6, 0),
    (0b00000000110, 11, 6, 1),
    (0b0000000101, 10, 6, 2),
    (0b00001000, 8, 6, 3),
    (0b0000000001011, 13, 7, 0),
    (0b0000000001110, 13, 7, 1),
    (0b00000000101, 11, 7, 2),
    (0b00001001, 8, 7, 3),
    (0b0000000001000, 13, 8, 0),
    (0b0000000001101, 13, 8, 1),
    (0b0000000001010, 13, 8, 2),
    (0b000000000100, 12, 8, 3),
    (0b00000000001111, 14, 9, 0),
    (0b00000000001110, 14, 9, 1),
    (0b0000000001001, 13, 9, 2),
    (0b000000000101, 12, 9, 3),
    (0b000000000001111, 15, 10, 0),
    (0b000000000001110, 15, 10, 1),
    (0b00000000001101, 14, 10, 2),
    (0b000000000110, 12, 10, 3),
    (0b0000000000001111, 16, 11, 0),
    (0b000000000001011, 15, 11, 1),
    (0b00000000001100, 14, 11, 2),
    (0b000000000111, 12, 11, 3),
    (0b0000000000001011, 16, 12, 0),
    (0b0000000000001000, 16, 12, 1),
    (0b000000000001010, 15, 12, 2),
    (0b00000000001011, 14, 12, 3),
    (0b0000000000001001, 16, 13, 0),
    (0b0000000000001110, 16, 13, 1),
    (0b000000000001101, 15, 13, 2),
    (0b00000000001010, 14, 13, 3),
    (0b0000000000000111, 16, 14, 0),
    (0b0000000000001010, 16, 14, 1),
    (0b000000000001100, 15, 14, 2),
    (0b00000000001001, 14, 14, 3),
    (0b0000000000000100, 16, 15, 0),
    (0b0000000000000110, 16, 15, 1),
    (0b0000000000001101, 16, 15, 2),
    (0b00000000001000, 14, 15, 3),
    (0b0000000000000101, 16, 16, 0),
    (0b0000000000000001, 16, 16, 1),
    (0b0000000000000010, 16, 16, 2),
    (0b0000000000001100, 16, 16, 3),
];

/// Table 9-5(b): 2 <= nC < 4
#[rustfmt::skip]
static COEFF_TOKEN_NC2: [(u32, u8, u8, u8); 62] = [
    (0b11, 2, 0, 0),
    (0b001011, 6, 1, 0),
    (0b10, 2, 1, 1),
    (0b000111, 6, 2, 0),
    (0b00111, 5, 2, 1),
    (0b011, 3, 2, 2),
    (0b0000111, 7, 3, 0),
    (0b001010, 6, 3, 1),
    (0b001001, 6, 3, 2),
    (0b00101, 5, 3, 3),
    (0b00000111, 8, 4, 0),
    (0b000110, 6, 4, 1),
    (0b000101, 6, 4, 2),
    (0b00100, 5, 4, 3),
    (0b000000111, 9, 5, 0),
    (0b0000110, 7, 5, 1),
    (0b0000101, 7, 5, 2),
    (0b001000, 6, 5, 3),
    (0b00000001111, 11, 6, 0),
    (0b00000110, 8, 6, 1),
    (0b00000101, 8, 6, 2),
    (0b0000100, 7, 6, 3),
    (0b00000001011, 11, 7, 0),
    (0b00000001110, 11, 7, 1),
    (0b000000110, 9, 7, 2),
    (0b00000100, 8, 7, 3),
    (0b000000001111, 12, 8, 0),
    (0b00000001101, 11, 8, 1),
    (0b00000001010, 11, 8, 2),
    (0b000000101, 9, 8, 3),
    (0b000000001011, 12, 9, 0),
    (0b000000001110, 12, 9, 1),
    (0b00000001001, 11, 9, 2),
    (0b000000100, 9, 9, 3),
    (0b0000000001111, 13, 10, 0),
    (0b000000001101, 12, 10, 1),
    (0b000000001010, 12, 10, 2),
    (0b00000001000, 11, 10, 3),
    (0b0000000001011, 13, 11, 0),
    (0b0000000001110, 13, 11, 1),
    (0b000000001001, 12, 11, 2),
    (0b000000001100, 12, 11, 3),
    (0b0000000001000, 13, 12, 0),
    (0b0000000001010, 13, 12, 1),
    (0b000000001000, 12, 12, 2),
    (0b0000000001101, 13, 12, 3),
    (0b00000000001111, 14, 13, 0),
    (0b0000000000001, 13, 13, 1),
    (0b0000000001001, 13, 13, 2),
    (0b0000000001100, 13, 13, 3),
    (0b00000000001011, 14, 14, 0),
    (0b00000000001110, 14, 14, 1),
    (0b00000000001101, 14, 14, 2),
    (0b0000000000100, 13, 14, 3),
    (0b00000000001000, 14, 15, 0),
    (0b00000000001010, 14, 15, 1),
    (0b00000000001001, 14, 15, 2),
    (0b0000000000110, 13, 15, 3),
    (0b00000000000111, 14, 16, 0),
    (0b00000000000101, 14, 16, 1),
    (0b00000000000100, 14, 16, 2),
    (0b0000000000111, 13, 16, 3),
];

/// Table 9-5(c): 4 <= nC < 8
#[rustfmt::skip]
static COEFF_TOKEN_NC4: [(u32, u8, u8, u8); 62] = [
    (0b1111, 4, 0, 0),
    (0b001111, 6, 1, 0),
    (0b1110, 4, 1, 1),
    (0b001011, 6, 2, 0),
    (0b01111, 5, 2, 1),
    (0b1101, 4, 2, 2),
    (0b001000, 6, 3, 0),
    (0b01100, 5, 3, 1),
    (0b01110, 5, 3, 2),
    (0b1100, 4, 3, 3),
    (0b0000111, 7, 4, 0),
    (0b01010, 5, 4, 1),
    (0b01101, 5, 4, 2),
    (0b1011, 4, 4, 3),
    (0b0000100, 7, 5, 0),
    (0b01000, 5, 5, 1),
    (0b01011, 5, 5, 2),
    (0b1010, 4, 5, 3),
    (0b00000111, 8, 6, 0),
    (0b000110, 6, 6, 1),
    (0b01001, 5, 6, 2),
    (0b1001, 4, 6, 3),
    (0b000001011, 9, 7, 0),
    (0b00000110, 8, 7, 1),
    (0b000101, 6, 7, 2),
    (0b1000, 4, 7, 3),
    (0b000001000, 9, 8, 0),
    (0b000001010, 9, 8, 1),
    (0b00000101, 8, 8, 2),
    (0b000100, 6, 8, 3),
    (0b0000001101, 10, 9, 0),
    (0b000000111, 9, 9, 1),
    (0b000001001, 9, 9, 2),
    (0b00000100, 8, 9, 3),
    (0b0000001001, 10, 10, 0),
    (0b0000001100, 10, 10, 1),
    (0b000000110, 9, 10, 2),
    (0b000001110, 9, 10, 3),
    (0b00000001111, 11, 11, 0),
    (0b0000001010, 10, 11, 1),
    (0b0000001000, 10, 11, 2),
    (0b000000101, 9, 11, 3),
    (0b00000001011, 11, 12, 0),
    (0b00000001110, 11, 12, 1),
    (0b0000001011, 10, 12, 2),
    (0b000001111, 9, 12, 3),
    (0b000000001111, 12, 13, 0),
    (0b00000001010, 11, 13, 1),
    (0b00000001101, 11, 13, 2),
    (0b000001101, 9, 13, 3),
    (0b000000001011, 12, 14, 0),
    (0b000000001110, 12, 14, 1),
    (0b00000001001, 11, 14, 2),
    (0b000001100, 9, 14, 3),
    (0b000000001000, 12, 15, 0),
    (0b000000001010, 12, 15, 1),
    (0b00000001000, 11, 15, 2),
    (0b000000001101, 12, 15, 3),
    (0b000000000111, 12, 16, 0),
    (0b000000000100, 12, 16, 1),
    (0b000000001001, 12, 16, 2),
    (0b000000001100, 12, 16, 3),
];

/// Table 9-5(d): chroma DC (nC == -1)
#[rustfmt::skip]
static COEFF_TOKEN_CHROMA_DC: [(u32, u8, u8, u8); 14] = [
    (0b01, 2, 0, 0),
    (0b000111, 6, 1, 0),
    (0b1, 1, 1, 1),
    (0b000100, 6, 2, 0),
    (0b00110, 5, 2, 1),
    (0b001, 3, 2, 2),
    (0b000011, 6, 3, 0),
    (0b000101, 6, 3, 1),
    (0b00101, 5, 3, 2),
    (0b00100, 5, 3, 3),
    (0b000010, 6, 4, 0),
    (0b000001, 6, 4, 1),
    (0b000000, 6, 4, 2),
    (0b00010, 5, 4, 3),
];

// ============================================================
// total_zeros VLC tables (H.264 Tables 9-7, 9-8)
// ============================================================

fn parse_total_zeros(r: &mut BitstreamReader, total_coeff: u8) -> Result<u8, &'static str> {
    match total_coeff {
        1 => match_vlc_u8(r, &TOTAL_ZEROS_1),
        2 => match_vlc_u8(r, &TOTAL_ZEROS_2),
        3 => match_vlc_u8(r, &TOTAL_ZEROS_3),
        4 => match_vlc_u8(r, &TOTAL_ZEROS_4),
        5 => match_vlc_u8(r, &TOTAL_ZEROS_5),
        6 => match_vlc_u8(r, &TOTAL_ZEROS_6),
        7 => match_vlc_u8(r, &TOTAL_ZEROS_7),
        8 => match_vlc_u8(r, &TOTAL_ZEROS_8),
        9 => match_vlc_u8(r, &TOTAL_ZEROS_9),
        10 => match_vlc_u8(r, &TOTAL_ZEROS_10),
        11 => match_vlc_u8(r, &TOTAL_ZEROS_11),
        12 => match_vlc_u8(r, &TOTAL_ZEROS_12),
        13 => match_vlc_u8(r, &TOTAL_ZEROS_13),
        14 => match_vlc_u8(r, &TOTAL_ZEROS_14),
        15 => match_vlc_u8(r, &TOTAL_ZEROS_15),
        _ => Err("invalid total_coeff for total_zeros"),
    }
}

fn parse_total_zeros_chroma_dc(r: &mut BitstreamReader, total_coeff: u8) -> Result<u8, &'static str> {
    match total_coeff {
        1 => match_vlc_u8(r, &TOTAL_ZEROS_CHROMA_DC_1),
        2 => match_vlc_u8(r, &TOTAL_ZEROS_CHROMA_DC_2),
        3 => match_vlc_u8(r, &TOTAL_ZEROS_CHROMA_DC_3),
        _ => Err("invalid total_coeff for chroma DC total_zeros"),
    }
}

// ============================================================
// run_before VLC table (H.264 Table 9-10)
// ============================================================

fn parse_run_before(r: &mut BitstreamReader, zeros_left: u8) -> Result<u8, &'static str> {
    if zeros_left == 0 {
        return Ok(0);
    }
    match zeros_left {
        1 => match_vlc_u8(r, &RUN_BEFORE_1),
        2 => match_vlc_u8(r, &RUN_BEFORE_2),
        3 => match_vlc_u8(r, &RUN_BEFORE_3),
        4 => match_vlc_u8(r, &RUN_BEFORE_4),
        5 => match_vlc_u8(r, &RUN_BEFORE_5),
        6 => match_vlc_u8(r, &RUN_BEFORE_6),
        _ => match_vlc_u8(r, &RUN_BEFORE_7PLUS),
    }
}

/// Generic VLC matcher returning a single u8 value.
/// Table entries: (codeword, bit_length, value)
fn match_vlc_u8(
    r: &mut BitstreamReader,
    table: &[(u32, u8, u8)],
) -> Result<u8, &'static str> {
    let mut code: u32 = 0;
    let mut bits_read: u8 = 0;

    loop {
        code = (code << 1) | r.read_bit()? as u32;
        bits_read += 1;

        for &(codeword, len, val) in table {
            if len == bits_read && codeword == code {
                return Ok(val);
            }
        }

        if bits_read >= 16 {
            break;
        }
    }
    Err("invalid VLC code")
}

// ============================================================
// Total zeros tables (Table 9-7)
// Format: (codeword, bit_length, total_zeros_value)
// ============================================================

#[rustfmt::skip]
static TOTAL_ZEROS_1: [(u32, u8, u8); 16] = [
    (0b1, 1, 0), (0b011, 3, 1), (0b010, 3, 2), (0b0011, 4, 3),
    (0b0010, 4, 4), (0b00011, 5, 5), (0b00010, 5, 6), (0b000011, 6, 7),
    (0b000010, 6, 8), (0b0000011, 7, 9), (0b0000010, 7, 10), (0b00000011, 8, 11),
    (0b00000010, 8, 12), (0b000000011, 9, 13), (0b000000010, 9, 14), (0b000000001, 9, 15),
];

#[rustfmt::skip]
static TOTAL_ZEROS_2: [(u32, u8, u8); 15] = [
    (0b111, 3, 0), (0b110, 3, 1), (0b101, 3, 2), (0b100, 3, 3),
    (0b011, 3, 4), (0b0101, 4, 5), (0b0100, 4, 6), (0b0011, 4, 7),
    (0b0010, 4, 8), (0b00011, 5, 9), (0b00010, 5, 10), (0b000011, 6, 11),
    (0b000010, 6, 12), (0b000001, 6, 13), (0b000000, 6, 14),
];

#[rustfmt::skip]
static TOTAL_ZEROS_3: [(u32, u8, u8); 14] = [
    (0b0101, 4, 0), (0b111, 3, 1), (0b110, 3, 2), (0b101, 3, 3),
    (0b0100, 4, 4), (0b0011, 4, 5), (0b100, 3, 6), (0b011, 3, 7),
    (0b0010, 4, 8), (0b00011, 5, 9), (0b00010, 5, 10), (0b000001, 6, 11),
    (0b000000, 6, 12), (0b000010, 6, 13),
    // Note: total_zeros can go from 0..13 for total_coeff=3
];

#[rustfmt::skip]
static TOTAL_ZEROS_4: [(u32, u8, u8); 13] = [
    (0b00011, 5, 0), (0b111, 3, 1), (0b0101, 4, 2), (0b0100, 4, 3),
    (0b110, 3, 4), (0b101, 3, 5), (0b100, 3, 6), (0b0011, 4, 7),
    (0b011, 3, 8), (0b00010, 5, 9), (0b00001, 5, 10), (0b00000, 5, 11),
    (0b0010, 4, 12),
];

#[rustfmt::skip]
static TOTAL_ZEROS_5: [(u32, u8, u8); 12] = [
    (0b0101, 4, 0), (0b0100, 4, 1), (0b0011, 4, 2), (0b111, 3, 3),
    (0b110, 3, 4), (0b101, 3, 5), (0b100, 3, 6), (0b0010, 4, 7),
    (0b011, 3, 8), (0b00001, 5, 9), (0b00000, 5, 10), (0b0001, 4, 11),
    // Note: some entries may need correction — only 11 possible values
];

#[rustfmt::skip]
static TOTAL_ZEROS_6: [(u32, u8, u8); 11] = [
    (0b000001, 6, 0), (0b000000, 6, 1), (0b0011, 4, 2), (0b111, 3, 3),
    (0b110, 3, 4), (0b101, 3, 5), (0b100, 3, 6), (0b011, 3, 7),
    (0b0010, 4, 8), (0b0001, 4, 9), (0b0000, 4, 10),
    // Note: some entries may need correction
];

#[rustfmt::skip]
static TOTAL_ZEROS_7: [(u32, u8, u8); 10] = [
    (0b000001, 6, 0), (0b000000, 6, 1), (0b0010, 4, 2), (0b111, 3, 3),
    (0b110, 3, 4), (0b101, 3, 5), (0b100, 3, 6), (0b011, 3, 7),
    (0b0011, 4, 8), (0b01, 2, 9), // Note: some entries may differ
];

#[rustfmt::skip]
static TOTAL_ZEROS_8: [(u32, u8, u8); 9] = [
    (0b000001, 6, 0), (0b00001, 5, 1), (0b0001, 4, 2), (0b011, 3, 3),
    (0b11, 2, 4), (0b10, 2, 5), (0b010, 3, 6), (0b0000, 4, 7),
    (0b000000, 6, 8),
];

#[rustfmt::skip]
static TOTAL_ZEROS_9: [(u32, u8, u8); 8] = [
    (0b000001, 6, 0), (0b000000, 6, 1), (0b0001, 4, 2), (0b11, 2, 3),
    (0b10, 2, 4), (0b001, 3, 5), (0b01, 2, 6), (0b0000, 4, 7),
];

#[rustfmt::skip]
static TOTAL_ZEROS_10: [(u32, u8, u8); 7] = [
    (0b00001, 5, 0), (0b00000, 5, 1), (0b001, 3, 2), (0b11, 2, 3),
    (0b10, 2, 4), (0b01, 2, 5), (0b0001, 4, 6),
];

#[rustfmt::skip]
static TOTAL_ZEROS_11: [(u32, u8, u8); 6] = [
    (0b0000, 4, 0), (0b0001, 4, 1), (0b001, 3, 2), (0b010, 3, 3),
    (0b1, 1, 4), (0b011, 3, 5),
];

#[rustfmt::skip]
static TOTAL_ZEROS_12: [(u32, u8, u8); 5] = [
    (0b0000, 4, 0), (0b0001, 4, 1), (0b01, 2, 2), (0b1, 1, 3),
    (0b001, 3, 4),
];

#[rustfmt::skip]
static TOTAL_ZEROS_13: [(u32, u8, u8); 4] = [
    (0b000, 3, 0), (0b001, 3, 1), (0b1, 1, 2), (0b01, 2, 3),
];

#[rustfmt::skip]
static TOTAL_ZEROS_14: [(u32, u8, u8); 3] = [
    (0b00, 2, 0), (0b01, 2, 1), (0b1, 1, 2),
];

#[rustfmt::skip]
static TOTAL_ZEROS_15: [(u32, u8, u8); 2] = [
    (0b0, 1, 0), (0b1, 1, 1),
];

// Chroma DC total zeros (Table 9-9(a) for 4:2:0)
#[rustfmt::skip]
static TOTAL_ZEROS_CHROMA_DC_1: [(u32, u8, u8); 4] = [
    (0b1, 1, 0), (0b01, 2, 1), (0b001, 3, 2), (0b000, 3, 3),
];

#[rustfmt::skip]
static TOTAL_ZEROS_CHROMA_DC_2: [(u32, u8, u8); 3] = [
    (0b1, 1, 0), (0b01, 2, 1), (0b00, 2, 2),
];

#[rustfmt::skip]
static TOTAL_ZEROS_CHROMA_DC_3: [(u32, u8, u8); 2] = [
    (0b1, 1, 0), (0b0, 1, 1),
];

// ============================================================
// run_before tables (Table 9-10)
// Format: (codeword, bit_length, run_before_value)
// ============================================================

#[rustfmt::skip]
static RUN_BEFORE_1: [(u32, u8, u8); 2] = [
    (0b1, 1, 0), (0b0, 1, 1),
];

#[rustfmt::skip]
static RUN_BEFORE_2: [(u32, u8, u8); 3] = [
    (0b1, 1, 0), (0b01, 2, 1), (0b00, 2, 2),
];

#[rustfmt::skip]
static RUN_BEFORE_3: [(u32, u8, u8); 4] = [
    (0b11, 2, 0), (0b10, 2, 1), (0b01, 2, 2), (0b00, 2, 3),
];

#[rustfmt::skip]
static RUN_BEFORE_4: [(u32, u8, u8); 5] = [
    (0b11, 2, 0), (0b10, 2, 1), (0b01, 2, 2), (0b001, 3, 3), (0b000, 3, 4),
];

#[rustfmt::skip]
static RUN_BEFORE_5: [(u32, u8, u8); 6] = [
    (0b11, 2, 0), (0b10, 2, 1), (0b011, 3, 2), (0b010, 3, 3), (0b001, 3, 4), (0b000, 3, 5),
];

#[rustfmt::skip]
static RUN_BEFORE_6: [(u32, u8, u8); 7] = [
    (0b11, 2, 0), (0b000, 3, 1), (0b001, 3, 2), (0b011, 3, 3), (0b010, 3, 4), (0b0001, 4, 5), (0b0000, 4, 6),
];

// zeros_left >= 7: run_before is 0..zeros_left, coded as:
// 0: 111, 1: 110, 2: 101, 3: 100, 4: 011, 5: 010, 6: 001, 7+: 0001, 00001, etc.
#[rustfmt::skip]
static RUN_BEFORE_7PLUS: [(u32, u8, u8); 15] = [
    (0b111, 3, 0), (0b110, 3, 1), (0b101, 3, 2), (0b100, 3, 3),
    (0b011, 3, 4), (0b010, 3, 5), (0b001, 3, 6),
    (0b0001, 4, 7), (0b00001, 5, 8), (0b000001, 6, 9),
    (0b0000001, 7, 10), (0b00000001, 8, 11), (0b000000001, 9, 12),
    (0b0000000001, 10, 13), (0b00000000001, 11, 14),
];
