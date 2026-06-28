use rust_h264::decoder::{Decoder, Frame};
use rust_h264::nal::parse_annex_b;
use rust_h264::sha256::sha256_hex;
use std::alloc::{GlobalAlloc, Layout, System};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

struct CountingAllocator;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static DEALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        DEALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

const STREAMS: &[&str] = &[
    "bench_720p_300f_ponly.h264",
    "bench_720p_300f_ponly_complex.h264",
    "bench_720p_300f_bframes.h264",
    "bench_720p_300f_bframes_complex.h264",
    "bench_1080p_100f.h264",
    "bench_1080p_100f_complex.h264",
    "bench_1080p_100f_ponly.h264",
];

#[derive(Clone)]
struct DecodeResult {
    frames: usize,
    digest: String,
    allocs: usize,
}

fn decode_once(data: &[u8]) -> DecodeResult {
    let nals = parse_annex_b(data);
    let before_allocs = ALLOCATIONS.load(Ordering::Relaxed);
    let mut decoder = Decoder::new();
    let mut frames = Vec::new();
    for nal in &nals {
        if let Some(frame) = decoder.decode_nal(nal).unwrap() {
            frames.push(frame);
        }
    }
    if let Some(frame) = decoder.flush() {
        frames.push(frame);
    }
    let allocs = ALLOCATIONS.load(Ordering::Relaxed) - before_allocs;

    frames.sort_by_key(|frame| frame.pic_order_cnt);
    DecodeResult {
        frames: frames.len(),
        digest: digest_frames(&frames),
        allocs,
    }
}

fn digest_frames(frames: &[Frame]) -> String {
    let mut output = Vec::new();
    for frame in frames {
        output.extend_from_slice(&frame.y);
        output.extend_from_slice(&frame.u);
        output.extend_from_slice(&frame.v);
    }
    sha256_hex(&output)
}

fn decode_timed(data: &[u8]) -> (DecodeResult, Duration) {
    let start = Instant::now();
    let result = decode_once(data);
    (result, start.elapsed())
}

fn has_tool(name: &str) -> bool {
    Command::new("which")
        .arg(name)
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn generate_missing_1080p_ponly(path: &str) -> bool {
    if !has_tool("ffmpeg") {
        return false;
    }

    Command::new("ffmpeg")
        .args([
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=1920x1080:rate=30:duration=3.33",
            "-frames:v",
            "100",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "23",
            "-x264opts",
            "bframes=0:ref=1:no-deblock:keyint=250:min-keyint=25:cabac=1",
            "-f",
            "h264",
            path,
        ])
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let warmup = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(3usize);
    let measured = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5usize);
    let testdata = format!("{}/testdata", env!("CARGO_MANIFEST_DIR"));
    let missing_ponly = format!("{testdata}/bench_1080p_100f_ponly.h264");

    if !std::path::Path::new(&missing_ponly).exists()
        && !generate_missing_1080p_ponly(&missing_ponly)
    {
        eprintln!("[MISSING] 1080p ponly cabac — generate with x264");
    }

    println!("stream | frames | decode_ms | fps | allocs | sha256(prefix)");
    println!("--- | ---: | ---: | ---: | ---: | ---");

    for stream in STREAMS {
        let path = format!("{testdata}/{stream}");
        let Ok(data) = std::fs::read(&path) else {
            println!("{stream} | MISSING | - | - | - | -");
            continue;
        };

        for _ in 0..warmup {
            let _ = decode_once(&data);
        }

        let mut best: Option<(DecodeResult, Duration)> = None;
        for _ in 0..measured {
            let current = decode_timed(&data);
            if best
                .as_ref()
                .map(|(_, best_elapsed)| current.1 < *best_elapsed)
                .unwrap_or(true)
            {
                best = Some(current);
            }
        }

        let (result, elapsed) = best.unwrap();
        let decode_ms = elapsed.as_secs_f64() * 1000.0;
        let fps = result.frames as f64 / elapsed.as_secs_f64();
        let prefix = &result.digest[..16];
        println!(
            "{stream} | {} | {:.3} | {:.2} | {} | {prefix}",
            result.frames, decode_ms, fps, result.allocs
        );
        println!("golden {stream}: {}", result.digest);
    }
}
