"""
SadTalker Optimized Local App - Pre-process face + voice once, generate videos from text instantly

Usage:
1. Run setup once: Pre-process face image + voice file
2. Generate videos: Enter text → uses cached face/voice → fast generation

Cost optimization:
- Face preprocessing: Once (saves ~5-10s GPU per generation)
- Voice caching: Once (no voice stripping needed)
- Fast generation: ~3-5x faster than full pipeline
"""

import os
import sys
import pickle
import shutil
import torch
import gradio as gr
import asyncio
import edge_tts
from pathlib import Path
from datetime import datetime
from pydub import AudioSegment
import numpy as np

# Fix numpy 2.x compatibility
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'int'):
    np.int = int

# Auto-detect BASE_DIR
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
possible_paths = [
    script_dir,
    os.path.join(script_dir, "SadTalker"),
    os.path.dirname(script_dir),
]

BASE_DIR = None
for path in possible_paths:
    if os.path.exists(os.path.join(path, "inference.py")):
        BASE_DIR = path
        break

if BASE_DIR is None:
    print("❌ Error: Could not find SadTalker directory!")
    sys.exit(1)

# Paths
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
RESULT_DIR = os.path.join(BASE_DIR, "results")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(os.path.join(ASSETS_DIR, "image"), exist_ok=True)
os.makedirs(os.path.join(ASSETS_DIR, "audio"), exist_ok=True)

# Cache files
FACE_CACHE_FILE = os.path.join(CACHE_DIR, "face_cache.pkl")
VOICE_CACHE_FILE = os.path.join(CACHE_DIR, "voice_cache.pkl")

# Default assets
DEFAULT_IMAGE = os.path.join(ASSETS_DIR, "image", "female-image-01.jpg")
DEFAULT_VOICE_MP3 = os.path.join(ASSETS_DIR, "audio", "female-voice-01.mp3")
DEFAULT_VOICE_WAV = os.path.join(ASSETS_DIR, "audio", "female-voice-01.wav")

print(f"📁 BASE_DIR: {BASE_DIR}")
print(f"📁 Cache: {CACHE_DIR}")
print(f"📁 Results: {RESULT_DIR}")
print(f"📁 Assets: {ASSETS_DIR}")

# Add to path
sys.path.insert(0, BASE_DIR)


def preprocess_and_cache_face(image_path: str, cache_id: str = "default"):
    """Pre-process face once and cache 3DMM coefficients."""
    print("🔄 Pre-processing face (this runs once)...")
    
    from src.utils.preprocess import CropAndExtract
    from src.utils.init_path import init_path
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Using device: {device}")
    
    sadtalker_paths = init_path(CHECKPOINT_DIR, os.path.join(BASE_DIR, 'src/config'), 256, False, 'full')
    preprocess_model = CropAndExtract(sadtalker_paths, device)
    
    # Extract face coefficients (expensive - done once)
    cache_frame_dir = os.path.join(CACHE_DIR, f"face_{cache_id}")
    os.makedirs(cache_frame_dir, exist_ok=True)
    
    first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(
        image_path, cache_frame_dir, 'full', source_image_flag=True, pic_size=256
    )
    
    if first_coeff_path is None:
        return None, "❌ Face detection failed. Use a clear front-facing face image."
    
    # Find landmarks file if it exists
    pic_name = os.path.splitext(os.path.split(image_path)[-1])[0]
    landmarks_path = os.path.join(cache_frame_dir, f"{pic_name}_landmarks.txt")
    
    # Cache results
    cache_data = {
        'first_coeff_path': first_coeff_path,
        'crop_pic_path': crop_pic_path,
        'crop_info': crop_info,
        'image_path': image_path,
        'landmarks_path': landmarks_path if os.path.exists(landmarks_path) else None,
        'cache_id': cache_id
    }
    
    with open(FACE_CACHE_FILE, 'wb') as f:
        pickle.dump(cache_data, f)
    
    # Cleanup
    del preprocess_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return cache_data, f"✅ Face pre-processed and cached!\n   Coefficients: {os.path.basename(first_coeff_path)}"


def load_face_cache():
    """Load cached face data."""
    if os.path.exists(FACE_CACHE_FILE):
        with open(FACE_CACHE_FILE, 'rb') as f:
            return pickle.load(f)
    return None


def preprocess_and_cache_voice(audio_path: str, cache_id: str = "default"):
    """Pre-process voice file (convert MP3→WAV if needed, cache path)."""
    print("🔄 Pre-processing voice file...")
    
    # Convert MP3 to WAV if needed
    if audio_path.endswith('.mp3'):
        wav_path = audio_path.replace('.mp3', '.wav')
        if not os.path.exists(wav_path):
            print(f"   Converting MP3 → WAV...")
            audio = AudioSegment.from_mp3(audio_path)
            audio.export(wav_path, format="wav")
        audio_path = wav_path
    
    # Cache voice file path
    cache_data = {
        'voice_path': audio_path,
        'cache_id': cache_id
    }
    
    with open(VOICE_CACHE_FILE, 'wb') as f:
        pickle.dump(cache_data, f)
    
    return cache_data, f"✅ Voice file cached!\n   File: {os.path.basename(audio_path)}"


def load_voice_cache():
    """Load cached voice data."""
    if os.path.exists(VOICE_CACHE_FILE):
        with open(VOICE_CACHE_FILE, 'rb') as f:
            return pickle.load(f)
    return None


async def text_to_speech_async(text: str, voice: str, out_path: str):
    """Generate speech from text using edge-tts."""
    mp3_path = out_path.replace(".wav", ".mp3")
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(mp3_path)
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(out_path, format="wav")
    if os.path.exists(mp3_path):
        os.remove(mp3_path)
    return out_path


# Global model cache (load once, reuse)
_model_cache = {}

def get_models():
    """Load models once and cache them."""
    global _model_cache
    
    if _model_cache:
        return _model_cache
    
    print("🔄 Loading models (first time only)...")
    from src.utils.init_path import init_path
    from src.test_audio2coeff import Audio2Coeff
    from src.facerender.animate import AnimateFromCoeff
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sadtalker_paths = init_path(CHECKPOINT_DIR, os.path.join(BASE_DIR, 'src/config'), 256, False, 'full')
    
    audio_to_coeff = Audio2Coeff(sadtalker_paths, device)
    animate_from_coeff = AnimateFromCoeff(sadtalker_paths, device)
    
    _model_cache = {
        'audio_to_coeff': audio_to_coeff,
        'animate_from_coeff': animate_from_coeff,
        'sadtalker_paths': sadtalker_paths,
        'device': device
    }
    
    return _model_cache


def generate_video_fast(text: str, use_cached_voice: bool = False):
    """FAST generation - bypasses inference.py, uses cached face directly."""
    import time
    start_time = time.time()
    
    # Load cached face
    face_cache = load_face_cache()
    if not face_cache:
        return None, "❌ No cached face found. Run Setup Mode first."
    
    # Generate or use audio
    ts = datetime.now().strftime("%Y_%m_%d_%H.%M.%S")
    audio_path = os.path.join(RESULT_DIR, f"audio_{ts}.wav")
    
    if use_cached_voice:
        voice_cache = load_voice_cache()
        if voice_cache and os.path.exists(voice_cache['voice_path']):
            shutil.copy(voice_cache['voice_path'], audio_path)
            print(f"✓ Using cached voice")
        else:
            return None, "❌ No cached voice found. Use TTS or run Setup."
    else:
        print("🔄 Generating speech from text...")
        voice_id = "en-US-JennyNeural"
        asyncio.run(text_to_speech_async(text.strip(), voice_id, audio_path))
    
    print(f"✓ Audio ready ({time.time() - start_time:.1f}s)")
    
    # Get models (cached after first load)
    models = get_models()
    audio_to_coeff = models['audio_to_coeff']
    animate_from_coeff = models['animate_from_coeff']
    device = models['device']
    
    # Create output directory
    gen_dir = os.path.join(RESULT_DIR, f"gen_{ts}")
    os.makedirs(gen_dir, exist_ok=True)
    
    print("🔄 Processing audio → coefficients...")
    # Step 1: Audio → coefficients (using cached face coeff)
    from src.generate_batch import get_data
    
    batch = get_data(
        face_cache['first_coeff_path'], 
        audio_path, 
        device, 
        ref_eyeblink_coeff_path=None, 
        still=True
    )
    
    coeff_path = audio_to_coeff.generate(batch, gen_dir, pose_style=0, ref_pose_coeff_path=None)
    print(f"✓ Coefficients generated ({time.time() - start_time:.1f}s)")
    
    # Step 2: Coefficients → video
    print("🔄 Generating video...")
    from src.generate_facerender_batch import get_facerender_data
    
    data = get_facerender_data(
        coeff_path, 
        face_cache['crop_pic_path'], 
        face_cache['first_coeff_path'], 
        audio_path,
        batch_size=2,
        input_yaw_list=None,
        input_pitch_list=None,
        input_roll_list=None,
        expression_scale=1.0,
        still_mode=True,
        preprocess='full',
        size=256
    )
    
    # Use enhancer=None to avoid OOM on low-RAM systems (full video is still produced)
    result = animate_from_coeff.generate(
        data, 
        gen_dir, 
        face_cache['image_path'], 
        face_cache['crop_info'],
        enhancer=None,  # set to 'gfpgan' if you have enough RAM for face enhancement
        background_enhancer=None,
        preprocess='full',
        img_size=256
    )
    
    # Move result to final location
    final_video = os.path.join(gen_dir, f"result_{ts}.mp4")
    if os.path.exists(result):
        shutil.move(result, final_video)
    else:
        final_video = result
    
    elapsed = time.time() - start_time
    print(f"✅ Complete! ({elapsed:.1f}s)")
    
    return final_video, f"✅ Generated in {elapsed:.1f}s: {os.path.basename(final_video)}"


def auto_setup_from_assets():
    """Automatically setup using default assets if they exist."""
    face_cache = load_face_cache()
    voice_cache = load_voice_cache()
    
    results = []
    errors = []
    
    # Setup face
    if os.path.exists(DEFAULT_IMAGE):
        if not face_cache:
            try:
                _, msg = preprocess_and_cache_face(DEFAULT_IMAGE, "female-01")
                results.append(msg)
            except Exception as e:
                errors.append(f"❌ Face setup failed: {str(e)}")
        else:
            results.append("✓ Face already cached")
    else:
        errors.append(f"⚠ Image not found: {DEFAULT_IMAGE}\n   Upload female-image-01.jpg to assets/image/")
    
    # Setup voice
    voice_file = None
    if os.path.exists(DEFAULT_VOICE_WAV):
        voice_file = DEFAULT_VOICE_WAV
    elif os.path.exists(DEFAULT_VOICE_MP3):
        voice_file = DEFAULT_VOICE_MP3
    
    if voice_file:
        if not voice_cache:
            try:
                _, msg = preprocess_and_cache_voice(voice_file, "female-01")
                results.append(msg)
            except Exception as e:
                errors.append(f"❌ Voice setup failed: {str(e)}")
        else:
            results.append("✓ Voice already cached")
    else:
        errors.append(f"⚠ Voice not found. Upload female-voice-01.mp3 to assets/audio/")
    
    output = "\n".join(results) if results else ""
    if errors:
        output += "\n\n" + "\n".join(errors)
    
    return output if output else "✅ Ready! Both face and voice are cached."


# Gradio UI
with gr.Blocks(title="SadTalker — Optimized Local") as demo:
    gr.Markdown(f"""
    # 🚀 SadTalker: Optimized Cached Setup (Local)
    
    **Pre-process face + voice once → Generate videos from text instantly**
    
    **📁 Cache:** `{CACHE_DIR}`  
    **📁 Results:** `{RESULT_DIR}`  
    **📁 Assets:** `{ASSETS_DIR}`
    """)
    
    with gr.Tabs():
        with gr.TabItem("1️⃣ Setup (Run Once)"):
            gr.Markdown("### Pre-process face + voice → Cache for fast generation")
            
            # Auto-setup from assets
            with gr.Row():
                auto_setup_btn = gr.Button("🚀 Auto-Setup from Assets", variant="primary", scale=2)
                auto_setup_status = gr.Textbox(label="Auto-Setup Status", interactive=False, lines=5)
            
            auto_setup_btn.click(fn=auto_setup_from_assets, outputs=[auto_setup_status])
            
            gr.Markdown("---\n### Or Manual Setup:")
            
            with gr.Row():
                with gr.Column():
                    setup_image = gr.Image(type="filepath", label="Face Image")
                    setup_cache_id = gr.Textbox(label="Face Cache ID", value="default")
                    setup_face_btn = gr.Button("Pre-process Face", variant="secondary")
                
                with gr.Column():
                    setup_voice = gr.Audio(type="filepath", label="Voice Audio File")
                    setup_voice_id = gr.Textbox(label="Voice Cache ID", value="default")
                    setup_voice_btn = gr.Button("Cache Voice", variant="secondary")
            
            setup_status = gr.Textbox(label="Setup Status", interactive=False, lines=3)
            
            def do_setup_face(image, cache_id):
                if not image:
                    return "Please upload a face image"
                image_path = image if isinstance(image, str) else image.get("path") or getattr(image, "name", None)
                _, msg = preprocess_and_cache_face(image_path, cache_id or "default")
                return msg
            
            def do_setup_voice(audio, cache_id):
                if not audio:
                    return "Please upload a voice audio file"
                audio_path = audio if isinstance(audio, str) else audio.get("path") or getattr(audio, "name", None)
                _, msg = preprocess_and_cache_voice(audio_path, cache_id or "default")
                return msg
            
            setup_face_btn.click(fn=do_setup_face, inputs=[setup_image, setup_cache_id], outputs=[setup_status])
            setup_voice_btn.click(fn=do_setup_voice, inputs=[setup_voice, setup_voice_id], outputs=[setup_status])
        
        with gr.TabItem("2️⃣ Generate (Fast)"):
            gr.Markdown("### Enter text → Generate video (uses cached face + voice)")
            
            gen_text = gr.Textbox(label="Text to speak", lines=4, placeholder="Enter the text for the avatar to read...")
            
            with gr.Row():
                gen_mode = gr.Radio(
                    choices=["Use TTS (Text-to-Speech)", "Use Cached Voice File"],
                    value="Use TTS (Text-to-Speech)",
                    label="Audio Source"
                )
                gen_btn = gr.Button("🚀 Generate Video", variant="primary", scale=2)
            
            gen_video = gr.Video(label="Output Video")
            gen_status = gr.Textbox(label="Status", interactive=False, lines=3)
            
            def do_generate(text, mode):
                if not text or not text.strip():
                    return None, "Please enter some text"
                
                use_cached = (mode == "Use Cached Voice File")
                video_path, status = generate_video_fast(text, use_cached_voice=use_cached)
                return video_path, status
            
            gen_btn.click(fn=do_generate, inputs=[gen_text, gen_mode], outputs=[gen_video, gen_status])
    
    gr.Markdown("""
    ### 💡 Usage:
    1. **Setup (once):** Upload face image + voice file → Click "Auto-Setup" or manual setup
    2. **Generate:** Enter text → Click "Generate Video" → Uses cached face/voice
    3. **Cost savings:** ~45% faster (face preprocessing skipped)
    """)

if __name__ == "__main__":
    print("\n🚀 Starting SadTalker Optimized Local App...")
    print(f"📁 Working directory: {BASE_DIR}")
    print(f"💻 CUDA available: {torch.cuda.is_available()}\n")
    
    demo.launch(
        debug=True,
        share=False,
        server_name="127.0.0.1",
        server_port=7860
    )
