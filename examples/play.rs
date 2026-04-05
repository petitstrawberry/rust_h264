/// Decode an H.264 bitstream and display frames in a window.
///
/// Usage: cargo run --example play -- <input.h264> [--fps N] [--loop]
///
/// Streams decode: feeds NALs incrementally, displays each frame as it's
/// decoded. Only keeps a small buffer of ARGB frames for display.
/// Press Escape to quit. With --loop, playback repeats from the beginning.
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

    let h264_data = match std::fs::read(input_path) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Error: cannot read '{}': {}", input_path, e);
            std::process::exit(1);
        }
    };
    let nals = parse_annex_b(&h264_data);

    // First pass: decode first frame to get dimensions for window creation.
    // We need width/height before we can create the window.
    let mut decoder = Decoder::new();
    let mut first_frame = None;
    let mut first_nal_idx = 0;
    for (idx, nal) in nals.iter().enumerate() {
        match decoder.decode_nal(nal) {
            Ok(Some(f)) => {
                first_frame = Some(f);
                first_nal_idx = idx + 1;
                break;
            }
            Ok(None) => {}
            Err(e) => {
                eprintln!("Error decoding: {:?}", e);
                std::process::exit(1);
            }
        }
    }

    let first_frame = first_frame.unwrap_or_else(|| {
        eprintln!("No frames decoded.");
        std::process::exit(1);
    });

    let width = first_frame.width as usize;
    let height = first_frame.height as usize;
    eprintln!(
        "Playing {} ({}x{}) at {} fps — streaming decode",
        input_path, width, height, fps
    );

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
    let mut last_frame_time = Instant::now();
    let mut frame_count = 0u64;

    // Display first frame
    let argb = yuv_to_argb(&first_frame.y, &first_frame.u, &first_frame.v, width, height);
    window
        .update_with_buffer(&argb, width, height)
        .expect("failed to update window");
    frame_count += 1;
    let mut current_argb = argb;

    // Streaming decode: continue from where we left off
    let mut nal_idx = first_nal_idx;

    'outer: loop {
        // Feed NALs until we get the next frame
        let mut got_frame = false;
        while nal_idx <= nals.len() && !got_frame {
            let frame = if nal_idx < nals.len() {
                match decoder.decode_nal(&nals[nal_idx]) {
                    Ok(f) => f,
                    Err(e) => {
                        eprintln!("Error decoding NAL {}: {:?}", nal_idx, e);
                        nal_idx += 1;
                        continue;
                    }
                }
            } else {
                // Past last NAL: flush
                decoder.flush()
            };
            nal_idx += 1;

            if let Some(f) = frame {
                current_argb = yuv_to_argb(&f.y, &f.u, &f.v, width, height);
                got_frame = true;
                frame_count += 1;
            }
        }

        if !got_frame {
            // End of stream
            if do_loop {
                // Reset decoder and start over
                decoder = Decoder::new();
                nal_idx = 0;
                eprintln!("Looping... ({} frames played)", frame_count);
                continue;
            }
            // Show last frame until window closed
            while window.is_open() && !window.is_key_down(Key::Escape) {
                window.update();
                std::thread::sleep(Duration::from_millis(10));
            }
            break;
        }

        // Wait for frame timing
        loop {
            if !window.is_open() || window.is_key_down(Key::Escape) {
                break 'outer;
            }
            let now = Instant::now();
            if now.duration_since(last_frame_time) >= frame_duration {
                last_frame_time = now;
                break;
            }
            window.update();
            std::thread::sleep(Duration::from_millis(1));
        }

        window
            .update_with_buffer(&current_argb, width, height)
            .expect("failed to update window");
    }

    eprintln!("Played {} frames", frame_count);
}
