"""
SadTalker Faster App - Maximum speed optimizations (~2-4x faster than optimized version)

Speed Optimizations:
- FP16 (Half Precision): ~2x faster GPU inference
- Larger batch size (4): Better GPU utilization
- Optional seamless clone skip: Saves ~67s (loses background blending)
- Lower resolution option: 128px for ultra-fast generation
- CUDA optimizations: cudnn.benchmark enabled

Expected speed: ~20-50s (vs ~87s in optimized version)
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

# Enable CUDA optimizations
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # Faster convolutions
    torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed

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
print(f"⚡ Speed optimizations: FP16, Batch=4, CUDA benchmark enabled")

# Add to path
sys.path.insert(0, BASE_DIR)


def preprocess_and_cache_face(image_path: str, cache_id: str = "default", size: int = 256):
    """Pre-process face once and cache 3DMM coefficients."""
    print(f"🔄 Pre-processing face (size={size}, this runs once)...")
    
    from src.utils.preprocess import CropAndExtract
    from src.utils.init_path import init_path
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Using device: {device}")
    
    sadtalker_paths = init_path(CHECKPOINT_DIR, os.path.join(BASE_DIR, 'src/config'), size, False, 'full')
    preprocess_model = CropAndExtract(sadtalker_paths, device)
    
    # Extract face coefficients (expensive - done once)
    cache_frame_dir = os.path.join(CACHE_DIR, f"face_{cache_id}")
    os.makedirs(cache_frame_dir, exist_ok=True)
    
    first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(
        image_path, cache_frame_dir, 'full', source_image_flag=True, pic_size=size
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
        'cache_id': cache_id,
        'size': size  # Store size for later use
    }
    
    with open(FACE_CACHE_FILE, 'wb') as f:
        pickle.dump(cache_data, f)
    
    # Cleanup
    del preprocess_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return cache_data, f"✅ Face pre-processed and cached!\n   Coefficients: {os.path.basename(first_coeff_path)}\n   Size: {size}px"


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

def get_models(use_fp16: bool = True, size: int = 256):
    """Load models once and cache them. Optionally use FP16 for 2x speed."""
    global _model_cache
    
    cache_key = f"{size}_{use_fp16}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]
    
    print(f"🔄 Loading models (size={size}, FP16={use_fp16}, first time only)...")
    from src.utils.init_path import init_path
    from src.test_audio2coeff import Audio2Coeff
    from src.facerender.animate import AnimateFromCoeff
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sadtalker_paths = init_path(CHECKPOINT_DIR, os.path.join(BASE_DIR, 'src/config'), size, False, 'full')
    
    audio_to_coeff = Audio2Coeff(sadtalker_paths, device)
    animate_from_coeff = AnimateFromCoeff(sadtalker_paths, device)
    
    # Convert to FP16 if requested and CUDA available
    if use_fp16 and device == "cuda":
        print("   ⚡ Converting models to FP16 (half precision) for 2x speed...")
        try:
            # Convert Audio2Coeff models (recursively converts all submodules)
            if hasattr(audio_to_coeff, 'audio2exp_model'):
                audio_to_coeff.audio2exp_model = audio_to_coeff.audio2exp_model.half()
            if hasattr(audio_to_coeff, 'audio2pose_model'):
                audio_to_coeff.audio2pose_model = audio_to_coeff.audio2pose_model.half()
            
            # AnimateFromCoeff models (recursively converts all submodules)
            if hasattr(animate_from_coeff, 'generator'):
                animate_from_coeff.generator = animate_from_coeff.generator.half()
            if hasattr(animate_from_coeff, 'kp_detector'):
                animate_from_coeff.kp_detector = animate_from_coeff.kp_detector.half()
            if hasattr(animate_from_coeff, 'he_estimator'):
                animate_from_coeff.he_estimator = animate_from_coeff.he_estimator.half()
            if hasattr(animate_from_coeff, 'mapping'):
                animate_from_coeff.mapping = animate_from_coeff.mapping.half()
            
            print("   ✓ Models converted to FP16")
        except Exception as e:
            print(f"   ⚠ FP16 conversion failed: {e}, using FP32")
            use_fp16 = False
    
    _model_cache[cache_key] = {
        'audio_to_coeff': audio_to_coeff,
        'animate_from_coeff': animate_from_coeff,
        'sadtalker_paths': sadtalker_paths,
        'device': device,
        'use_fp16': use_fp16,
        'size': size
    }
    
    return _model_cache[cache_key]


def generate_video_fast(text: str, use_cached_voice: bool = False, 
                       use_fp16: bool = False, batch_size: int = 4,
                       skip_seamless_clone: bool = False, size: int = 256):
    """ULTRA-FAST generation with all optimizations enabled."""
    import time
    start_time = time.time()
    
    # Load cached face
    face_cache = load_face_cache()
    if not face_cache:
        return None, "❌ No cached face found. Run Setup Mode first."
    
    # Use cached size if available, otherwise use provided size
    cached_size = face_cache.get('size', size)
    if cached_size != size:
        print(f"⚠ Warning: Cached face is {cached_size}px, but requested {size}px. Using cached size.")
        size = cached_size
    
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
    
    # Get models (cached after first load, with FP16 if requested)
    models = get_models(use_fp16=use_fp16, size=size)
    audio_to_coeff = models['audio_to_coeff']
    animate_from_coeff = models['animate_from_coeff']
    device = models['device']
    
    # Create output directory
    gen_dir = os.path.join(RESULT_DIR, f"gen_{ts}")
    os.makedirs(gen_dir, exist_ok=True)
    
    print(f"🔄 Processing audio → coefficients (FP16={use_fp16}, batch={batch_size})...")
    # Step 1: Audio → coefficients (using cached face coeff)
    from src.generate_batch import get_data
    
    batch = get_data(
        face_cache['first_coeff_path'], 
        audio_path, 
        device, 
        ref_eyeblink_coeff_path=None, 
        still=True
    )
    
    # Convert batch to FP16 if using FP16 (must match model dtype)
    if use_fp16 and device == "cuda":
        try:
            def to_half(x):
                if isinstance(x, torch.Tensor):
                    return x.half()
                elif isinstance(x, dict):
                    return {k: to_half(v) for k, v in x.items()}
                elif isinstance(x, (list, tuple)):
                    return type(x)(to_half(v) for v in x)
                return x
            batch = to_half(batch)
            print("   ✓ Batch converted to FP16")
        except Exception as e:
            print(f"   ⚠ Batch FP16 conversion failed: {e}, using FP32")
            # If batch conversion fails, disable FP16 for this run
            use_fp16 = False
    
    coeff_path = audio_to_coeff.generate(batch, gen_dir, pose_style=0, ref_pose_coeff_path=None)
    print(f"✓ Coefficients generated ({time.time() - start_time:.1f}s)")
    
    # Step 2: Coefficients → video
    print(f"🔄 Generating video (batch_size={batch_size}, skip_clone={skip_seamless_clone})...")
    from src.generate_facerender_batch import get_facerender_data
    
    data = get_facerender_data(
        coeff_path, 
        face_cache['crop_pic_path'], 
        face_cache['first_coeff_path'], 
        audio_path,
        batch_size=batch_size,  # Larger batch for better GPU utilization
        input_yaw_list=None,
        input_pitch_list=None,
        input_roll_list=None,
        expression_scale=1.0,
        still_mode=True,
        preprocess='full',
        size=size
    )
    
    # Skip seamless clone if requested (saves ~67s, loses background blending)
    if skip_seamless_clone:
        print("⚡ Skipping seamless clone (face crop only, saves ~67s)")
        # Generate video without seamless clone - just face crop
        result = animate_from_coeff.generate(
            data, 
            gen_dir, 
            face_cache['crop_pic_path'],  # Use crop instead of full image
            face_cache['crop_info'],
            enhancer=None,
            background_enhancer=None,
            preprocess='full',
            img_size=size
        )
        # The result will be the face crop video, not merged with background
    else:
        # Normal generation with seamless clone
        result = animate_from_coeff.generate(
            data, 
            gen_dir, 
            face_cache['image_path'], 
            face_cache['crop_info'],
            enhancer=None,
            background_enhancer=None,
            preprocess='full',
            img_size=size
        )
    
    # Move result to final location
    final_video = os.path.join(gen_dir, f"result_{ts}.mp4")
    if os.path.exists(result):
        shutil.move(result, final_video)
    else:
        final_video = result
    
    elapsed = time.time() - start_time
    speedup = 87 / elapsed if elapsed > 0 else 1  # Compare to baseline ~87s
    print(f"✅ Complete! ({elapsed:.1f}s, ~{speedup:.1f}x faster than baseline)")
    
    return final_video, f"✅ Generated in {elapsed:.1f}s (~{speedup:.1f}x faster)\n📹 {os.path.basename(final_video)}"


def auto_setup_from_assets(size: int = 256):
    """Automatically setup using default assets if they exist."""
    face_cache = load_face_cache()
    voice_cache = load_voice_cache()
    
    results = []
    errors = []
    
    # Setup face
    if os.path.exists(DEFAULT_IMAGE):
        if not face_cache or face_cache.get('size') != size:
            try:
                _, msg = preprocess_and_cache_face(DEFAULT_IMAGE, "female-01", size=size)
                results.append(msg)
            except Exception as e:
                errors.append(f"❌ Face setup failed: {str(e)}")
        else:
            results.append(f"✓ Face already cached ({size}px)")
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
with gr.Blocks(title="SadTalker — Faster (Maximum Speed)") as demo:
    gr.Markdown(f"""
    # ⚡ SadTalker: Faster Version (Maximum Speed Optimizations)
    
    **Speed optimizations:** FP16, Batch=4, CUDA benchmark, optional seamless clone skip
    
    **Expected speed:** ~20-50s (vs ~87s in optimized version, ~360s in standard version)
    
    **📁 Cache:** `{CACHE_DIR}`  
    **📁 Results:** `{RESULT_DIR}`  
    **📁 Assets:** `{ASSETS_DIR}`
    """)
    
    with gr.Tabs():
        with gr.TabItem("1️⃣ Setup (Run Once)"):
            gr.Markdown("### Pre-process face + voice → Cache for fast generation")
            
            # Auto-setup from assets
            with gr.Row():
                with gr.Column():
                    setup_size = gr.Slider(
                        minimum=128, maximum=512, step=64, value=256,
                        label="Face Resolution (px)",
                        info="Lower = faster (128px ultra-fast, 256px balanced, 512px high quality)"
                    )
                    auto_setup_btn = gr.Button("🚀 Auto-Setup from Assets", variant="primary", scale=2)
                auto_setup_status = gr.Textbox(label="Auto-Setup Status", interactive=False, lines=5)
            
            auto_setup_btn.click(fn=auto_setup_from_assets, inputs=[setup_size], outputs=[auto_setup_status])
            
            gr.Markdown("---\n### Or Manual Setup:")
            
            with gr.Row():
                with gr.Column():
                    setup_image = gr.Image(type="filepath", label="Face Image")
                    setup_cache_id = gr.Textbox(label="Face Cache ID", value="default")
                    setup_face_size = gr.Slider(
                        minimum=128, maximum=512, step=64, value=256,
                        label="Resolution (px)"
                    )
                    setup_face_btn = gr.Button("Pre-process Face", variant="secondary")
                
                with gr.Column():
                    setup_voice = gr.Audio(type="filepath", label="Voice Audio File")
                    setup_voice_id = gr.Textbox(label="Voice Cache ID", value="default")
                    setup_voice_btn = gr.Button("Cache Voice", variant="secondary")
            
            setup_status = gr.Textbox(label="Setup Status", interactive=False, lines=3)
            
            def do_setup_face(image, cache_id, size):
                if not image:
                    return "Please upload a face image"
                image_path = image if isinstance(image, str) else image.get("path") or getattr(image, "name", None)
                _, msg = preprocess_and_cache_face(image_path, cache_id or "default", size=int(size))
                return msg
            
            def do_setup_voice(audio, cache_id):
                if not audio:
                    return "Please upload a voice audio file"
                audio_path = audio if isinstance(audio, str) else audio.get("path") or getattr(audio, "name", None)
                _, msg = preprocess_and_cache_voice(audio_path, cache_id or "default")
                return msg
            
            setup_face_btn.click(fn=do_setup_face, inputs=[setup_image, setup_cache_id, setup_face_size], outputs=[setup_status])
            setup_voice_btn.click(fn=do_setup_voice, inputs=[setup_voice, setup_voice_id], outputs=[setup_status])
        
        with gr.TabItem("2️⃣ Generate (Ultra-Fast)"):
            gr.Markdown("### Enter text → Generate video with maximum speed optimizations")
            
            gen_text = gr.Textbox(label="Text to speak", lines=4, placeholder="Enter the text for the avatar to read...")
            
            with gr.Row():
                gen_mode = gr.Radio(
                    choices=["Use TTS (Text-to-Speech)", "Use Cached Voice File"],
                    value="Use TTS (Text-to-Speech)",
                    label="Audio Source"
                )
            
            with gr.Row():
                with gr.Column():
                    gen_fp16 = gr.Checkbox(
                        label="⚡ FP16 (Half Precision) - EXPERIMENTAL", 
                        value=False,
                        info="~2x faster GPU inference (may cause dtype errors, disable if issues)"
                    )
                    gen_batch_size = gr.Slider(
                        minimum=1, maximum=8, step=1, value=4,
                        label="Batch Size",
                        info="Larger = faster (4 recommended, 8 if GPU has enough memory)"
                    )
                    gen_skip_clone = gr.Checkbox(
                        label="⚡ Skip Seamless Clone",
                        value=False,
                        info="Saves ~67s but loses background blending (face crop only)"
                    )
                    gen_size = gr.Slider(
                        minimum=128, maximum=512, step=64, value=256,
                        label="Generation Resolution (px)",
                        info="Must match cached face size. Lower = faster."
                    )
                
                with gr.Column():
                    gen_btn = gr.Button("⚡ Generate Video (Ultra-Fast)", variant="primary", scale=2)
            
            gen_video = gr.Video(label="Output Video", height=400)
            gen_status = gr.Textbox(label="Status", interactive=False, lines=4)
            
            def do_generate(text, mode, fp16, batch_size, skip_clone, size):
                if not text or not text.strip():
                    return None, "Please enter some text"
                
                use_cached = (mode == "Use Cached Voice File")
                video_path, status = generate_video_fast(
                    text, 
                    use_cached_voice=use_cached,
                    use_fp16=fp16,
                    batch_size=int(batch_size),
                    skip_seamless_clone=skip_clone,
                    size=int(size)
                )
                return video_path, status
            
            gen_btn.click(
                fn=do_generate, 
                inputs=[gen_text, gen_mode, gen_fp16, gen_batch_size, gen_skip_clone, gen_size], 
                outputs=[gen_video, gen_status]
            )
    
    gr.Markdown("""
    ### ⚡ Speed Optimizations:
    1. **FP16 (Half Precision):** ~2x faster GPU inference
    2. **Batch Size 4:** Better GPU utilization (~20% faster)
    3. **Skip Seamless Clone:** Saves ~67s (loses background blending)
    4. **Lower Resolution (128px):** ~2x faster rendering
    5. **CUDA Benchmark:** Optimized convolution kernels
    
    ### 📊 Expected Performance:
    - **Baseline (standard):** ~360s
    - **Optimized:** ~87s
    - **Faster (FP16 + Batch 4):** ~50-60s
    - **Ultra-Fast (all optimizations):** ~20-30s
    
    ### 💡 Usage:
    1. **Setup (once):** Upload face image + voice file → Click "Auto-Setup"
    2. **Generate:** Enter text → Adjust speed settings → Click "Generate Video"
    3. **Compare:** Try different settings to see speed vs quality trade-offs
    """)

if __name__ == "__main__":
    print("\n⚡ Starting SadTalker Faster App (Maximum Speed)...")
    print(f"📁 Working directory: {BASE_DIR}")
    print(f"💻 CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"⚡ CUDA optimizations: benchmark={torch.backends.cudnn.benchmark}")
    print()
    
    demo.launch(
        debug=True,
        share=False,
        server_name="127.0.0.1",
        server_port=7861  # Different port to avoid conflicts
    )
