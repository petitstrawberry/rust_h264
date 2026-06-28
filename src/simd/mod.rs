//! SIMD backend dispatch for motion-compensation kernels.
//!
//! Two-layer model: a portable SIMD backend (`portable-simd` feature,
//! lowering to the target's native SIMD via `core::simd`) and a scalar
//! reference/fallback. No hand-written arch-specific backend is maintained.

#[cfg(feature = "portable-simd")]
mod portable;
#[cfg(not(feature = "portable-simd"))]
pub(crate) mod scalar;

#[cfg(feature = "portable-simd")]
pub use portable::*;
#[cfg(not(feature = "portable-simd"))]
pub use scalar::*;

#[cfg(test)]
mod tests {
    use alloc::format;
    use alloc::string::String;
    use alloc::vec::Vec;

    use crate::decoder::Decoder;
    use crate::nal::parse_annex_b;
    use crate::sha256::sha256_hex;

    const GOLDEN_STREAMS: &[(&str, &str)] = &[
        (
            "bench_720p_300f_ponly",
            "540b6d538b170f21b3c26b120299c34d3d7b71b7fdb653533643d738247d7658",
        ),
        (
            "bench_720p_300f_ponly_complex",
            "4b5cce0664658bf777832652d6d56b727e9a96d88e14e193c409309771e8f525",
        ),
        (
            "bench_720p_300f_bframes",
            "0d5d059d8d8632a33109e29ba3f78d94577dacec03ba7b97b052a6ce7908f54a",
        ),
        (
            "bench_720p_300f_bframes_complex",
            "28572d2b4b26bfc882e9e190d461ef360d3332edcf67730a6303a34fa0f062f0",
        ),
        (
            "bench_1080p_100f",
            "6a0486521aba40c2b3457d91f376de2b7f308be4fd396f2c4947468ac7734872",
        ),
        (
            "bench_1080p_100f_complex",
            "9181000ebd22fc2aebbed242937649251bc85cf1aa55a0c404e8feb3d083d9ba",
        ),
        (
            "bench_1080p_100f_ponly",
            "20b52c0de018ea316525119b703489828f471035a226f49ac9f3050c45ad1de3",
        ),
    ];

    fn decode_stream_digest(name: &str) -> String {
        let h264_path = format!("{}/testdata/{}.h264", env!("CARGO_MANIFEST_DIR"), name);
        let h264_data = std::fs::read(&h264_path)
            .unwrap_or_else(|e| panic!("failed to read {}: {}", h264_path, e));
        let nals = parse_annex_b(&h264_data);
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
        frames.sort_by_key(|frame| frame.pic_order_cnt);

        let mut output = Vec::new();
        for frame in &frames {
            output.extend_from_slice(&frame.y);
            output.extend_from_slice(&frame.u);
            output.extend_from_slice(&frame.v);
        }
        sha256_hex(&output)
    }

    #[test]
    #[cfg(not(feature = "portable-simd"))]
    fn golden_scalar_corpus_digests() {
        for &(name, expected) in GOLDEN_STREAMS {
            assert_eq!(
                decode_stream_digest(name),
                expected,
                "digest mismatch for {name}"
            );
        }
    }

    #[test]
    #[cfg(feature = "portable-simd")]
    fn golden_portable_matches_scalar() {
        for &(name, expected) in GOLDEN_STREAMS {
            assert_eq!(
                decode_stream_digest(name),
                expected,
                "portable digest mismatch for {name}"
            );
        }
    }
}
