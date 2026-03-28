/// Decode an H.264 bitstream and write all decoded frames as raw YUV420 in display order.
///
/// Usage: cargo run --example dump_frames -- <input.h264> [output.yuv]
///
/// If no output path is given, replaces the .h264 extension with .yuv.
/// Frames are sorted by POC (display order) within each IDR period.
use rust_h264::decoder::Decoder;
use rust_h264::nal::{parse_annex_b, NalUnitType};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: dump_frames <input.h264> [output.yuv]");
        std::process::exit(1);
    }
    let input_path = &args[1];
    let output_path = if args.len() >= 3 {
        args[2].clone()
    } else {
        input_path.replace(".h264", ".yuv")
    };

    let h264_data = std::fs::read(input_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {}", input_path, e));
    let nals = parse_annex_b(&h264_data);
    let mut decoder = Decoder::new();

    // Track IDR boundaries so we can sort within each period.
    // Each frame gets (idr_index, poc, decode_order) for sorting.
    let mut idr_count: u32 = 0;
    let mut frames: Vec<(u32, i32, usize, rust_h264::decoder::Frame)> = Vec::new();
    let mut decode_order: usize = 0;

    for nal in &nals {
        if nal.nal_unit_type == NalUnitType::SliceIdr {
            idr_count += 1;
        }
        match decoder.decode_nal(nal) {
            Ok(Some(f)) => {
                eprintln!(
                    "Decoded frame {}: poc={} {}x{}",
                    decode_order, f.pic_order_cnt, f.width, f.height
                );
                frames.push((idr_count, f.pic_order_cnt, decode_order, f));
                decode_order += 1;
            }
            Ok(None) => {}
            Err(e) => {
                eprintln!("Error decoding frame: {:?}", e);
                std::process::exit(1);
            }
        }
    }

    // Sort by (idr_period, poc) for display order within each IDR period.
    // Decode order is used as a stable tiebreaker for equal POCs.
    frames.sort_by_key(|&(idr, poc, order, _)| (idr, poc, order));

    let mut output = Vec::new();
    for (_, _, _, frame) in &frames {
        output.extend_from_slice(&frame.y);
        output.extend_from_slice(&frame.u);
        output.extend_from_slice(&frame.v);
    }
    std::fs::write(&output_path, &output)
        .unwrap_or_else(|e| panic!("failed to write {}: {}", output_path, e));
    eprintln!(
        "Wrote {} frames ({} bytes) to {}",
        frames.len(),
        output.len(),
        output_path
    );
}
