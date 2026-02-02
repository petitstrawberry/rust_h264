# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A pure Rust H.264 video decoder library. Aims to be a standalone, portable software H.264 decoder (unlike OpenH264 which only supports baseline profile, or FFmpeg's decoder which isn't available as a separate library). Part of the broader rust_media ecosystem.

## Build Commands

This is a Rust project using Cargo:

- **Build:** `cargo build`
- **Test:** `cargo test`
- **Run single test:** `cargo test <test_name>`
- **Lint:** `cargo clippy`
- **Format:** `cargo fmt`
- **Check:** `cargo check`

## Milestones

1. Get simple decoder test case working
2. Finish implementation of decoder
3. Compare performance of decoder against ffmpeg

## Design Decisions

- **Input format:** Annex B bytestream (start code delimited), not AVCC (length-prefixed). Callers must provide raw Annex B NAL units.
- **Streaming API:** The decoder API is streaming — callers feed NAL units incrementally and receive decoded frames as they become available. No requirement to buffer an entire stream upfront.
- **Performance:** The decoder should be fast. Prefer efficient algorithms, minimize allocations, and avoid unnecessary copies. Performance relative to ffmpeg's software decoder is a key benchmark.

## Status

Project is in early stages — SPS, PPS, and SEI parsing implemented. Slice decoding not yet started.
