# Demo Guide: TinyML Speech Emotion Recognition on UNIHIKER K10

This guide helps you create and share a clean demo of the project. You can use it for a short video, blogpost, or slides.

## What to show (suggested 60–120 seconds)

1) Intro (5–10 s)
- Show the UNIHIKER K10 board and mention “ESP32‑S3 + on‑board mic + TFT + local CNN”.

2) Boot and IP (10–15 s)
- Power on the board; show the TFT displaying the assigned IP.
- Open the IP in a browser on the same network; show the Tailwind dashboard loading.

3) Live inference (30–60 s)
- Speak a few sample phrases with different emotions.
- Let the camera capture both the TFT bars and the web dashboard updating.
- Point out the highest‑confidence emotion and cumulative counters.

4) Wrap up (5–10 s)
- Mention that the model runs fully on‑device and the web UI is served from the board.

## Optional screenshots to include in README/blog/slides

- `images/tft_bars.png`: TFT screen showing the bar chart and IP.
- `images/web_dashboard.png`: Tailwind dashboard with bars and counters.

(Place images under `docs/images/` and reference them relatively.)

## Tips for a smooth recording

- Use a quiet room; keep the board close to your mouth.
- If the dashboard seems slow, wait for an update tick (1.5 s cadence) or temporarily reduce background network load.
- If you don’t want to reveal your SSID in a video, crop or blur the TFT IP line.

## Link your demo here

- Video: <ADD_LINK_HERE>
- Blogpost: <ADD_LINK_HERE>
- Slides: <ADD_LINK_HERE>

