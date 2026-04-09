# rust_h264

While working on rust_media, it was found that there isn't any sufficiently good open source h264 decoder. There is openH264, but it is limited to baseline h264. ffmpeg has its own h264 decoder but it isn't split out as a library.

Hence, the idea is to attempt to create an open source h264 decoder.
Yes, most devices have hardware h264 decoder, but if we want to be truly portable, then software implementation of h264 decoder is needed.

## Design

- **Input:** Both Annex B (start code delimited `00 00 00 01` / `00 00 01`) and AVCC (length-prefixed, used inside MP4/MKV containers) bitstreams are supported. The decoder itself accepts `NalUnit` values; the choice of parser determines the input format.
- **Streaming:** The decoder exposes a streaming API. NAL units are fed incrementally and decoded frames are emitted as they become available.
- **Performance:** The decoder aims to be fast, with performance relative to ffmpeg's software H.264 decoder as the target benchmark.

## Usage

```rust
use rust_h264::decoder::Decoder;
use rust_h264::nal::parse_annex_b;

let h264_data = std::fs::read("input.h264").unwrap();
let nals = parse_annex_b(&h264_data);
let mut decoder = Decoder::new();

for nal in &nals {
    match decoder.decode_nal(nal) {
        Ok(Some(frame)) => {
            // `frame` is a decoded YUV420 picture:
            //   frame.y, frame.u, frame.v  — pixel planes
            //   frame.width, frame.height  — dimensions
            //   frame.pic_order_cnt        — display order index
        }
        Ok(None) => {} // NAL consumed, no frame ready yet (e.g. SPS/PPS)
        Err(e) => eprintln!("decode error: {:?}", e),
    }
}
// Flush the last buffered frame
if let Some(frame) = decoder.flush() {
    // handle final frame
}
```

### Important: frame ordering

**`decode_nal` returns frames in decode order, not display order.** With
B-frames, the decoder must buffer reference frames before it can decode
the B-frames that depend on them. This means the output order differs
from the intended display order.

To display frames correctly, sort them by `pic_order_cnt` (POC). If the
stream has multiple IDR boundaries (GOPs), you must also track IDR
boundaries to avoid mixing frames from different GOPs:

```rust
use rust_h264::nal::NalUnitType;

let mut idr_count: u32 = 0;
let mut frames = Vec::new();

for nal in &nals {
    let is_idr = nal.nal_unit_type == NalUnitType::SliceIdr;

    if let Ok(Some(frame)) = decoder.decode_nal(nal) {
        // Push with CURRENT idr_count — this frame belongs to the
        // previous picture, before the IDR boundary
        frames.push((idr_count, frame));
    }

    // Increment AFTER decode_nal, because decode_nal returns the
    // PREVIOUS frame when it sees a new picture header. If you
    // increment before, the last B-frame of the old GOP gets tagged
    // with the new GOP's count and sorts incorrectly.
    if is_idr {
        idr_count += 1;
    }
}
if let Some(frame) = decoder.flush() {
    frames.push((idr_count, frame));
}

// Sort by (GOP, POC) for display order
frames.sort_by_key(|(idr, f)| (*idr, f.pic_order_cnt));
```

### Common pitfall: IDR count timing

The most common mistake is incrementing `idr_count` **before** calling
`decode_nal`. This causes the last frame of each GOP to be placed after
the next IDR in display order, resulting in a visible glitch at every
scene cut.

**Wrong:**
```rust
if nal.nal_unit_type == NalUnitType::SliceIdr {
    idr_count += 1;  // BUG: too early
}
let frame = decoder.decode_nal(nal)?;
// frame belongs to the OLD GOP but gets the NEW idr_count
```

**Correct:**
```rust
let frame = decoder.decode_nal(nal)?;
// Push frame with current idr_count first
if nal.nal_unit_type == NalUnitType::SliceIdr {
    idr_count += 1;  // After the previous frame is handled
}
```

### AVCC input (MP4/MKV containers)

For length-prefixed bitstreams from MP4/MKV containers, use `parse_avcc_config`
for the `avcC` configuration box and `parse_avcc` for each sample. The decoder
itself is unchanged — only the framing parser differs.

```rust
use rust_h264::decoder::Decoder;
use rust_h264::nal::{parse_avcc, parse_avcc_config};

// Get the avcC box payload from your MP4 demuxer
let config = parse_avcc_config(&avcc_box_payload).unwrap();
let mut decoder = Decoder::new();

// Feed SPS/PPS once at startup (they live in the avcC box, not in samples)
for nal in config.sps_nals.iter().chain(config.pps_nals.iter()) {
    decoder.decode_nal(nal).unwrap();
}

// For each sample (MP4 chunk), parse and decode its NALs
for sample_data in mp4_samples {
    for nal in parse_avcc(&sample_data, config.length_size) {
        if let Ok(Some(frame)) = decoder.decode_nal(&nal) {
            // handle frame (apply same display-order sorting as Annex B)
        }
    }
}
```

`length_size` is taken from the `avcC` box (typically 4) and matches the
`lengthSizeMinusOne + 1` field. The AVCC NAL payload is identical to Annex B
(same NAL header, same RBSP, same emulation prevention handling).

## Tools

### Player

Decode and display an H.264 bitstream in a window:

```
cargo run --example play -- input.h264 [--fps 30] [--loop]
```

- `--fps N` — set playback frame rate (default: 30)
- `--loop` — loop playback continuously
- Press Escape to quit

### Dump frames

Decode an H.264 bitstream to raw YUV420 output:

```
cargo run --example dump_frames -- input.h264 [output.yuv]
```

Frames are written in display order (sorted by POC).

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
