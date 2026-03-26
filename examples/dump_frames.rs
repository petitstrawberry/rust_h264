/// Decode an H.264 bitstream and write all decoded frames as raw YUV420 in display order.
///
/// Usage: cargo run --example dump_frames -- <input.h264> [output.yuv]
///
/// If no output path is given, replaces the .h264 extension with .yuv.
/// Frames are sorted by POC (display order) before writing.
use rust_h264::decoder::Decoder;
use rust_h264::nal::parse_annex_b;

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
    let mut frames = Vec::new();
    for nal in &nals {
        match decoder.decode_nal(nal) {
            Ok(Some(f)) => {
                eprintln!(
                    "Decoded frame {}: poc={} {}x{}",
                    frames.len(),
                    f.pic_order_cnt,
                    f.width,
                    f.height
                );
                frames.push(f);
            }
            Ok(None) => {}
            Err(e) => {
                eprintln!("Error decoding frame: {:?}", e);
                std::process::exit(1);
            }
        }
    }

    // Sort by POC for display order
    frames.sort_by_key(|f| f.pic_order_cnt);

    let mut output = Vec::new();
    for frame in &frames {
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
