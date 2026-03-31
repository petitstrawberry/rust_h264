/// Decode an H.264 bitstream and display frames in a window.
///
/// Usage: cargo run --example play -- <input.h264> [--fps N] [--loop]
///
/// Decodes all frames first, sorts by display order, then plays them
/// in a window at the specified frame rate (default: 30 fps).
/// Press Escape to quit. With --loop, playback repeats continuously.
use minifb::{Key, Window, WindowOptions};
use rust_h264::decoder::Decoder;
use rust_h264::nal::{parse_annex_b, NalUnitType};
use std::time::{Duration, Instant};

/// Convert YUV420 frame to ARGB pixel buffer for display.
fn yuv_to_argb(y: &[u8], u: &[u8], v: &[u8], width: usize, height: usize) -> Vec<u32> {
    let cw = width / 2;
    let mut argb = vec![0u32; width * height];
    for row in 0..height {
        for col in 0..width {
            let y_val = y[row * width + col] as i32;
            let u_val = u[(row / 2) * cw + col / 2] as i32 - 128;
            let v_val = v[(row / 2) * cw + col / 2] as i32 - 128;

            let r = (y_val + ((v_val * 359 + 128) >> 8)).clamp(0, 255) as u32;
            let g = (y_val - ((u_val * 88 + v_val * 183 + 128) >> 8)).clamp(0, 255) as u32;
            let b = (y_val + ((u_val * 454 + 128) >> 8)).clamp(0, 255) as u32;

            argb[row * width + col] = (r << 16) | (g << 8) | b;
        }
    }
    argb
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: play <input.h264> [--fps N] [--loop]");
        std::process::exit(1);
    }
    let input_path = &args[1];

    let mut fps = 30.0f64;
    let mut do_loop = false;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--fps" => {
                i += 1;
                fps = args[i].parse().unwrap_or(30.0);
            }
            "--loop" => do_loop = true,
            _ => eprintln!("Unknown option: {}", args[i]),
        }
        i += 1;
    }

    // Decode all frames
    eprintln!("Decoding {}...", input_path);
    let h264_data = std::fs::read(input_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {}", input_path, e));
    let nals = parse_annex_b(&h264_data);
    let mut decoder = Decoder::new();

    let mut idr_count: u32 = 0;
    let mut frames: Vec<(u32, i32, usize, rust_h264::decoder::Frame)> = Vec::new();
    let mut decode_order: usize = 0;

    for nal in &nals {
        if nal.nal_unit_type == NalUnitType::SliceIdr {
            idr_count += 1;
        }
        match decoder.decode_nal(nal) {
            Ok(Some(f)) => {
                frames.push((idr_count, f.pic_order_cnt, decode_order, f));
                decode_order += 1;
            }
            Ok(None) => {}
            Err(e) => {
                eprintln!("Error decoding: {:?}", e);
                break;
            }
        }
    }

    if let Some(f) = decoder.flush() {
        frames.push((idr_count, f.pic_order_cnt, decode_order, f));
        let _ = decode_order;
    }

    if frames.is_empty() {
        eprintln!("No frames decoded.");
        std::process::exit(1);
    }

    // Sort by display order
    frames.sort_by_key(|&(idr, poc, order, _)| (idr, poc, order));

    let width = frames[0].3.width as usize;
    let height = frames[0].3.height as usize;
    eprintln!(
        "Decoded {} frames, {}x{}, playing at {} fps",
        frames.len(),
        width,
        height,
        fps
    );

    // Convert all frames to ARGB
    let argb_frames: Vec<Vec<u32>> = frames
        .iter()
        .map(|(_, _, _, f)| yuv_to_argb(&f.y, &f.u, &f.v, width, height))
        .collect();

    // Create window
    let scale = if width <= 128 && height <= 128 {
        4
    } else if width <= 320 && height <= 240 {
        2
    } else {
        1
    };

    let mut window = Window::new(
        &format!("rust_h264 — {} ({}x{})", input_path, width, height),
        width * scale,
        height * scale,
        WindowOptions {
            resize: true,
            scale_mode: minifb::ScaleMode::AspectRatioStretch,
            ..WindowOptions::default()
        },
    )
    .expect("failed to create window");

    let frame_duration = Duration::from_secs_f64(1.0 / fps);
    let mut frame_idx = 0;
    let mut last_frame_time = Instant::now();

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let now = Instant::now();
        if now.duration_since(last_frame_time) >= frame_duration {
            window
                .update_with_buffer(&argb_frames[frame_idx], width, height)
                .expect("failed to update window");

            frame_idx += 1;
            if frame_idx >= argb_frames.len() {
                if do_loop {
                    frame_idx = 0;
                } else {
                    // Show last frame until window is closed
                    frame_idx = argb_frames.len() - 1;
                }
            }
            last_frame_time = now;
        } else {
            window.update();
            std::thread::sleep(Duration::from_millis(1));
        }
    }
}
