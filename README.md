# rust_h264

While working on rust_media, it was found that there isn't any sufficiently good open source h264 decoder. There is openH264, but it is limited to baseline h264. ffmpeg has its own h264 decoder but it isn't split out as a library.

Hence, the idea is to attempt to create an open source h264 decoder.
Yes, most devices have hardware h264 decoder, but if we want to be truly portable, then software implementation of h264 decoder is needed.

## Design

- **Input:** Annex B bytestream format (start code delimited `00 00 00 01` / `00 00 01`). AVCC (length-prefixed) format is not supported — callers must convert to Annex B before feeding data to the decoder.
- **Streaming:** The decoder exposes a streaming API. NAL units are fed incrementally and decoded frames are emitted as they become available.
- **Performance:** The decoder aims to be fast, with performance relative to ffmpeg's software H.264 decoder as the target benchmark.

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
