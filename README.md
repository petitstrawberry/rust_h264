# rust_h264

While working on rust_media, it was found that there isn't any sufficiently good open source h264 decoder. There is openH264, but it is limited to baseline h264. ffmpeg has its own h264 decoder but it isn't split out as a library.

Hence, the idea is to attempt to create an open source h264 decoder.
Yes, most devices have hardware h264 decoder, but if we want to be truly portable, then software implementation of h264 decoder is needed.
