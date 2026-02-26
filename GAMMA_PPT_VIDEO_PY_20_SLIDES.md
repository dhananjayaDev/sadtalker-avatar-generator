# 20-Slide Gamma.ai PPT Guide: video.py Tech Stack & Real-Time Avatar

Use this outline to build your presentation in Gamma.ai. Each section = one slide (title + bullet points you can paste or expand).

---

## Slide 1 — Title
**Real-Time Lip-Sync Avatar: Tech Stack & Architecture**  
*video.py — SadTalker-Based Talking Head Overlay*

- Subtitle: From text to live talking avatar in one pipeline
- Your name / date / event

---

## Slide 2 — What Problem Does It Solve?
**One Sentence:** Overlay animated mouth movements onto an existing video so a person in the video “speaks” your text in real time.

- **Not** full video generation — we keep the original video and only replace the mouth region
- Use case: virtual presenter, avatar kiosk, interactive demo
- Outcome: User types text → avatar speaks it with lip-sync in a window or web UI

---

## Slide 3 — High-Level Flow (The Big Picture)
**Three stages:**

1. **Setup (once):** Extract face from video → build a library of mouth shapes (visemes)
2. **Runtime:** Text → speech (TTS) + text → viseme sequence → play video and blend the right mouth onto each frame
3. **Output:** Live window (OpenCV) or streaming in browser (Gradio)

- Everything runs on one codebase: `video.py` + SadTalker + edge-tts + OpenCV + Gradio

---

## Slide 4 — Tech Stack Overview
**Core technologies:**

| Layer        | Technology        | Role |
|-------------|-------------------|------|
| **Face / 3D** | SadTalker (PyTorch) | Face detection, 3DMM coefficients, mouth animation |
| **Speech**   | edge-tts           | Text → audio (WAV) |
| **Phonetics**| phonemizer / rules | Text → phonemes → visemes |
| **Video I/O**| OpenCV (cv2)       | Read video, blend patches, display, write MP4 |
| **UI**      | Gradio             | Web UI, streaming image, text input |
| **Audio play** | pygame / ffplay  | Play WAV in background |

- Python 3, NumPy, pickle for caching

---

## Slide 5 — SadTalker: The Face Engine
**What SadTalker does in video.py:**

- **CropAndExtract:** Detects face on reference frame, crops it, extracts 3DMM coefficients (identity, expression, pose)
- **Audio2Coeff:** Predicts expression coefficients from audio (used to build viseme frames)
- **AnimateFromCoeff:** Renders face image from coefficients → we crop the mouth from that image
- **face_alignment (68 pts):** Precise mouth bounding box from landmarks (stored as normalized fractions)

- Result: Resolution-independent mouth placement on the original video

---

## Slide 6 — edge-tts: Text to Speech
**Role:** Turn user text into natural speech (WAV).

- Microsoft Edge TTS API (no local heavy TTS model)
- Voices: e.g. en-US-JennyNeural, en-US-GuyNeural, en-GB-SoniaNeural
- Flow: `text → MP3 → convert to WAV` (pydub)
- WAV is used for: (1) playback, (2) duration/timing, (3) silence detection for closed mouth (viseme “M”)

---

## Slide 7 — Phonemes & Visemes
**Why we need them:** Audio alone doesn’t tell us “which mouth shape” per frame; we map text → phonemes → visemes → mouth images.

- **Phoneme:** Smallest sound unit (e.g. HH, EH, L, OW for “hello”)
- **Viseme:** Visual mouth shape (e.g. H, Eh, L, Oh) — we have 22 types + silence (M) + optional blink
- **Mapping:** `PHONEME_TO_VISEME` dictionary in video.py (ARPAbet / IPA → viseme code)
- **Source of phonemes:** phonemizer (espeak) or rule-based fallback

---

## Slide 8 — Viseme Library (Precomputed Mouth Patches)
**Built once in setup:**

- For each viseme type (Aa, Eh, M, W, …): generate short TTS phrase → SadTalker renders a face frame → crop mouth region → save as PNG (e.g. `mouth_vid_Aa.png`)
- Stored in `cache/visemes/`; paths in `video_viseme_library.pkl`
- At runtime we only **look up** the right PNG and blend it — no per-frame 3D rendering

- **22 mouth visemes + 1 blink (eye patch)** = fast, deterministic lip-sync

---

## Slide 9 — OpenCV: Video In/Out and Blending
**Responsibilities:**

- **Input:** `cv2.VideoCapture(video_path)` — read frame by frame; loop by resetting to frame 0
- **Blending:** Mouth (and optional eye) patch resized to mouth bbox, then blended with **elliptical soft mask** (smoothstep) to avoid hard edges
- **Display:** `cv2.imshow()` for CLI; for Gradio, frame is written to a shared buffer and streamed as image
- **Export:** `cv2.VideoWriter` for temp video; ffmpeg merges with WAV → final MP4

---

## Slide 10 — Caching: Face + Visemes
**What we cache (so setup runs once):**

- **face_video_cache.pkl:** crop_info, paths to first_coeff and crop_pic, reference image path, **mouth_lm_frac** (normalized mouth bbox from 68-point landmarks)
- **video_viseme_library.pkl:** mapping viseme name → path to full viseme image (we mainly use pre-cropped mouth PNGs from disk)
- **cache/visemes/*.png:** Mouth (and blink) patches

- New video or new person → run `--setup` again

---

## Slide 11 — Setup Pipeline (Step by Step)
**Sequence when you run `python video.py --setup`:**

1. Extract reference frame from video (e.g. middle frame) → save as `*_ref_frame.png`
2. Run SadTalker CropAndExtract → get coeffs, crop, crop_info
3. Run face_alignment on reference → compute mouth bbox → save as mouth_lm_frac in face cache
4. For each viseme: TTS phrase → Audio2Coeff → AnimateFromCoeff → pick frame → crop mouth → save PNG
5. Save face cache and viseme library to disk

- One-time cost (~minutes); then runtime is lightweight

---

## Slide 12 — Runtime Pipeline (CLI Mode)
**When you run `python video.py --text "Hello"`:**

1. Load face cache + viseme library; open video with OpenCV
2. Run edge-tts: text → WAV; compute audio duration and silence periods
3. Text → phonemes → visemes; spread visemes over time (frames per viseme)
4. Optional: build blink schedule (when to show “closed eye” patch)
5. Phase 1: Play raw video (idle) for a short time
6. Phase 2: Start audio playback; each frame: pick current viseme → blend mouth patch → show (and optionally save)
7. Phase 3: Idle again until user quits (or stop when saving)

---

## Slide 13 — Real-Time Avatar: What “Real Time” Means
**No pre-rendering of the full video:**

- Video is read **frame by frame** and displayed immediately
- TTS is generated **once per sentence** (fast with edge-tts)
- Viseme choice is a **time-based lookup** (current time → which viseme) + blend of pre-made patches
- Audio plays in a **background thread** (pygame/ffplay) while the main loop updates the image

- “Real time” = live display and audio, with minimal delay after user submits text

---

## Slide 14 — Threading: Gradio UI Mode
**Two main threads:**

1. **TTS worker (daemon):** Reads (text, voice) from queue → edge-tts → WAV → phonemes → visemes → silence/blink data → pushes “payload” to second queue
2. **Avatar thread (daemon):** Reads video in a loop; if no payload → show raw frame (“Listening…”); when payload arrives → play WAV and for each frame blend viseme mouth → write result to `_current_frame`

- Gradio’s streaming image reads `_current_frame` at ~30 fps → user sees continuous avatar

---

## Slide 15 — Blending: Elliptical Mask
**Why it matters:** A rectangular patch would look like a sticker; we want a soft mouth region.

- **Ellipse mask:** 1 at center, smooth falloff to 0 at edges (smoothstep)
- **Per-pixel alpha:** `alpha = mask * strength` (e.g. 0.7–0.85)
- **Blend:** `frame_region = (1 - alpha) * frame_region + alpha * patch`
- Parameters: softness (~0.4), strength (~0.7–0.85) — tunable for crisp vs soft look

---

## Slide 16 — Silence & Smoothing
**Silence:** During quiet parts of the WAV (dBFS below threshold), we force viseme to **M** (closed mouth).

- **Smoothing:** When viseme changes, we can blend previous and current mouth patch for 1–2 frames (smoothstep) to avoid popping
- **Frames per viseme (fpv):** How many frames one viseme lasts; derived from audio duration and number of visemes (clamped for natural speed)

---

## Slide 17 — Gradio UI Layout
**What the user sees:**

- **Left:** Streaming avatar (gr.Image with streaming=True) — either “Listening…” (idle) or lip-sync (speaking)
- **Right:** Text box (“What should the avatar say?”), voice dropdown, “Speak” button, status
- **Collapsed:** Setup accordion (video path, frame index, face size, Run Setup)
- **Flow:** User types → clicks Speak → TTS worker runs in background → avatar switches to speaking when payload is ready

---

## Slide 18 — CLI vs UI
**CLI (`video.py --text "..."`):**

- Single run: idle → speak → idle; optional `--save out.mp4`
- Good for: demos, batch tests, debugging

**UI (`video.py --ui`):**

- Persistent avatar; user can send multiple phrases without restarting
- Good for: live demos, kiosks, interactive presentations

- Same tech stack and same real-time avatar logic in both

---

## Slide 19 — Summary: Tech Stack One-Liner
**“Video.py drives a fixed video as a real-time talking avatar by overlaying pre-generated viseme mouth patches, synced to edge-tts and a text→phoneme→viseme pipeline, with optional Gradio UI using a TTS worker and an avatar thread.”**

- **Stack in one line:** SadTalker (face/3DMM) + edge-tts (TTS) + phonemizer/rules (visemes) + OpenCV (video + blend) + Gradio (streaming UI) + pygame/ffplay (audio)

---

## Slide 20 — Next Steps & Resources
**Try it yourself:**

- `python video.py --setup` then `python video.py --text "Your text"` or `python video.py --ui`
- **Docs in repo:** `VIDEO_PY_ARCHITECTURE.md`, this guide
- **Extend:** Different voices, other languages (phoneme mapping), different base videos

- **Takeaway:** Real-time avatar = smart caching (viseme library) + lightweight runtime (lookup + blend) + async TTS and video threads

---

## Tips for Gamma.ai

1. Use **“Outline” or “Paste outline”** and paste the 20 slide titles + 2–3 bullets per slide from above.
2. Choose a **tech/startup** theme so code and architecture look good.
3. Add a **simple diagram** on slides 3, 4, 11, 12, 14 (flow/threading) — Gamma can generate from short descriptions like “flow from text to TTS to viseme to blended frame”.
4. Keep **Slide 4 (tech stack)** as a clean table; Gamma handles tables well.
5. Use **Slide 19** as the “one slide” recap for Q&A.

You can copy-paste each slide block into Gamma’s outline or split into 20 slides manually.
