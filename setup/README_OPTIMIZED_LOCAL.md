# SadTalker Optimized Local App

**Pre-process face + voice once → Generate videos from text instantly**

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install edge-tts pydub gradio
# (Other dependencies should already be installed from requirements.txt)
```

### 2. Place your assets
```
SadTalker/
├── assets/
│   ├── image/
│   │   └── female-image-01.jpg  ← Your face image
│   └── audio/
│       └── female-voice-01.mp3   ← Your voice file
```

### 3. Run setup (once)
```bash
python setup_optimized.py
```
This will:
- Pre-process face image → Extract 3DMM coefficients → Cache
- Pre-process voice file → Convert MP3→WAV → Cache
- **Takes ~10-15 seconds (one-time)**

### 4. Run the optimized app
```bash
python sadtalker_optimized_local.py
```

Open browser: `http://127.0.0.1:7860`

## 📋 Usage

### Setup Tab (Run Once)
1. **Auto-Setup:** Click "🚀 Auto-Setup from Assets" 
   - Automatically processes `assets/image/female-image-01.jpg` and `assets/audio/female-voice-01.mp3`
   
2. **Manual Setup:** Upload face image + voice file separately

### Generate Tab (Fast)
1. Enter text in the text box
2. Choose audio source:
   - **Use TTS** - Generates speech from text (default)
   - **Use Cached Voice** - Uses your uploaded voice file directly
3. Click "🚀 Generate Video"
4. Video appears in ~10-15 seconds (vs ~20-25s without caching)

## 💰 Cost Optimization

**Before (per generation):**
- Face detection: ~5s GPU
- 3DMM extraction: ~5s GPU ⬅️ **SKIPPED with cache**
- Audio processing: ~2s GPU
- Video generation: ~10s GPU
- **Total: ~22s GPU time**

**After (first time - setup):**
- Face detection: ~5s GPU
- 3DMM extraction: ~5s GPU
- Voice caching: ~1s CPU
- **Total: ~11s GPU time (one-time)**

**After (subsequent generations):**
- Face detection: ~2s GPU (still runs but faster with cached landmarks)
- 3DMM extraction: **SKIPPED** (uses cached coeff) ⬅️ **~5s saved**
- Audio processing: ~2s GPU
- Video generation: ~10s GPU
- **Total: ~14s GPU time (~36% faster)**

## 📁 File Structure

```
SadTalker/
├── assets/
│   ├── image/
│   │   └── female-image-01.jpg
│   └── audio/
│       └── female-voice-01.mp3
├── cache/
│   ├── face_cache.pkl          ← Cached face data
│   ├── voice_cache.pkl          ← Cached voice path
│   └── face_female-01/          ← Cached face coefficients
│       └── female-image-01.mat
├── results/                     ← Generated videos
├── sadtalker_optimized_local.py ← Main app
└── setup_optimized.py           ← Setup script
```

## 🔧 How It Works

1. **Face Caching:**
   - Extracts 3DMM coefficients from face image
   - Saves to `cache/face_cache.pkl`
   - Copies coeff file to expected location before inference
   - `inference.py` detects existing coeff → skips expensive extraction

2. **Voice Caching:**
   - Converts MP3 → WAV (if needed)
   - Stores path in `cache/voice_cache.pkl`
   - Can use cached voice directly or generate TTS from text

3. **Fast Generation:**
   - Uses cached face coefficients (skips 3DMM extraction)
   - Uses TTS or cached voice file
   - Generates video with lip sync

## 🎯 Features

- ✅ **One-time setup** - Pre-process face + voice once
- ✅ **Fast generation** - ~36% faster than full pipeline
- ✅ **Text-to-speech** - Generate speech from text (edge-tts)
- ✅ **Voice file support** - Use your own voice file
- ✅ **Gradio UI** - Easy-to-use web interface
- ✅ **Local only** - No cloud/Colab needed

## 📝 Notes

- Face detection still runs (but faster with cached landmarks)
- 3DMM extraction is fully skipped (biggest time saver)
- Cache persists between runs
- To change face/voice, run setup again

## 🐛 Troubleshooting

**"No cached face found"**
- Run setup first: `python setup_optimized.py` or use Setup tab

**"Face detection failed"**
- Use a clear, front-facing face image
- Ensure face is well-lit and visible

**"Voice file not found"**
- Upload `female-voice-01.mp3` to `assets/audio/`
- Or use TTS mode instead
