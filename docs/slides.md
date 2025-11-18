# Slides (Speaker Notes)

Title: TinyML Speech Emotion Recognition on UNIHIKER K10

1. Motivation (1 slide)
- On-device speech emotion inferences for privacy and latency
- TinyML: quantized CNN + Mel filterbank features

2. Hardware (1 slide)
- UNIHIKER K10: ESP32‑S3, ES7210 codec, TFT 240×320
- 16 kHz mono audio capture via I2S

3. Model (2 slides)
- Edge Impulse pipeline: MFE → compact CNN → int8 quantization
- Metrics and constraints (memory, latency)

4. Firmware (2 slides)
- Inference loop, TFT bar chart with counts and IP overlay
- Wi‑Fi + HTTP server: Tailwind dashboard and `/data` JSON

5. Demo (1 slide)
- Live inferences; top label + cumulative counters

6. References (1 slide)
- Cite the attached TinyML paper validating MFE + quantized CNN for MCU SER

7. Links (1 slide)
- GitHub, video, blogpost
