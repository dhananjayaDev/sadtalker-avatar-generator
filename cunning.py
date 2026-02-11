"""
SadTalker Cunning App - Ultra-fast generation using pre-rendered template video

Cunning Optimization:
- Pre-render ONE template video with lip movements (once)
- For new text: Only generate audio → Replace audio track in template video
- Skips face rendering + seamless cloning for each generation
- Expected speed: ~1-3s (vs ~87s in optimized, ~360s in standard)

Trade-off: Lip movements may not perfectly match new text (uses template movements)
"""

import os
import sys
import pickle
import shutil
import torch
import gradio as gr
import asyncio
import edge_tts
import subprocess
from pathlib import Path
from datetime import datetime
from pydub import AudioSegment
import numpy as np
import cv2

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
TEMPLATE_VIDEO_FILE = os.path.join(CACHE_DIR, "template_video.mp4")
TEMPLATE_AUDIO_FILE = os.path.join(CACHE_DIR, "template_audio.wav")

# Default assets
DEFAULT_IMAGE = os.path.join(ASSETS_DIR, "image", "female-image-01.jpg")
DEFAULT_VOICE_MP3 = os.path.join(ASSETS_DIR, "audio", "female-voice-01.mp3")
DEFAULT_VOICE_WAV = os.path.join(ASSETS_DIR, "audio", "female-voice-01.wav")

print(f"📁 BASE_DIR: {BASE_DIR}")
print(f"📁 Cache: {CACHE_DIR}")
print(f"📁 Results: {RESULT_DIR}")
print(f"📁 Assets: {ASSETS_DIR}")
print(f"🎭 Cunning mode: Template video + audio replacement")

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
    
    cache_frame_dir = os.path.join(CACHE_DIR, f"face_{cache_id}")
    os.makedirs(cache_frame_dir, exist_ok=True)
    
    first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(
        image_path, cache_frame_dir, 'full', source_image_flag=True, pic_size=size
    )
    
    if first_coeff_path is None:
        return None, "❌ Face detection failed. Use a clear front-facing face image."
    
    pic_name = os.path.splitext(os.path.split(image_path)[-1])[0]
    landmarks_path = os.path.join(cache_frame_dir, f"{pic_name}_landmarks.txt")
    
    cache_data = {
        'first_coeff_path': first_coeff_path,
        'crop_pic_path': crop_pic_path,
        'crop_info': crop_info,
        'image_path': image_path,
        'landmarks_path': landmarks_path if os.path.exists(landmarks_path) else None,
        'cache_id': cache_id,
        'size': size
    }
    
    with open(FACE_CACHE_FILE, 'wb') as f:
        pickle.dump(cache_data, f)
    
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
    
    if audio_path.endswith('.mp3'):
        wav_path = audio_path.replace('.mp3', '.wav')
        if not os.path.exists(wav_path):
            print(f"   Converting MP3 → WAV...")
            audio = AudioSegment.from_mp3(audio_path)
            audio.export(wav_path, format="wav")
        audio_path = wav_path
    
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


def generate_template_video(template_text: str = None, template_audio_path: str = None, size: int = 256):
    """Generate template video once - this will be reused for all future generations."""
    import time
    start_time = time.time()
    
    print("🎭 Generating template video (this runs once)...")
    
    face_cache = load_face_cache()
    if not face_cache:
        return None, "❌ No cached face found. Run Setup Mode first."
    
    # Use template text or audio
    if template_audio_path and os.path.exists(template_audio_path):
        audio_path = template_audio_path
        print(f"   Using provided audio: {os.path.basename(audio_path)}")
    elif template_text:
        print(f"   Generating audio from template text: '{template_text[:50]}...'")
        ts = datetime.now().strftime("%Y_%m_%d_%H.%M.%S")
        audio_path = os.path.join(CACHE_DIR, f"template_audio_{ts}.wav")
        voice_id = "en-US-JennyNeural"
        asyncio.run(text_to_speech_async(template_text, voice_id, audio_path))
    else:
        # Default: Use a phoneme-rich text that covers common mouth movements
        default_text = "The quick brown fox jumps over the lazy dog. Hello world, how are you today? This is a template video for lip synchronization."
        print(f"   Using default phoneme-rich text")
        audio_path = os.path.join(CACHE_DIR, "template_audio_default.wav")
        voice_id = "en-US-JennyNeural"
        asyncio.run(text_to_speech_async(default_text, voice_id, audio_path))
    
    # Generate video using the optimized pipeline (from faster.py logic)
    from src.utils.init_path import init_path
    from src.test_audio2coeff import Audio2Coeff
    from src.facerender.animate import AnimateFromCoeff
    from src.generate_batch import get_data
    from src.generate_facerender_batch import get_facerender_data
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sadtalker_paths = init_path(CHECKPOINT_DIR, os.path.join(BASE_DIR, 'src/config'), size, False, 'full')
    
    audio_to_coeff = Audio2Coeff(sadtalker_paths, device)
    animate_from_coeff = AnimateFromCoeff(sadtalker_paths, device)
    
    gen_dir = os.path.join(CACHE_DIR, "template_gen")
    os.makedirs(gen_dir, exist_ok=True)
    
    print("   Processing audio → coefficients...")
    batch = get_data(
        face_cache['first_coeff_path'], 
        audio_path, 
        device, 
        ref_eyeblink_coeff_path=None, 
        still=True
    )
    
    coeff_path = audio_to_coeff.generate(batch, gen_dir, pose_style=0, ref_pose_coeff_path=None)
    
    print("   Generating video...")
    data = get_facerender_data(
        coeff_path, 
        face_cache['crop_pic_path'], 
        face_cache['first_coeff_path'], 
        audio_path,
        batch_size=4,  # Use batch 4 for speed
        input_yaw_list=None,
        input_pitch_list=None,
        input_roll_list=None,
        expression_scale=1.0,
        still_mode=True,
        preprocess='full',
        size=size
    )
    
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
    
    # Save template video and audio
    if os.path.exists(result):
        shutil.copy(result, TEMPLATE_VIDEO_FILE)
        shutil.copy(audio_path, TEMPLATE_AUDIO_FILE)
        elapsed = time.time() - start_time
        print(f"✅ Template video generated! ({elapsed:.1f}s)")
        return TEMPLATE_VIDEO_FILE, f"✅ Template video ready ({elapsed:.1f}s)\n📹 {os.path.basename(TEMPLATE_VIDEO_FILE)}\n🎵 {os.path.basename(TEMPLATE_AUDIO_FILE)}"
    else:
        return None, "❌ Template video generation failed"


def replace_audio_in_video(video_path: str, new_audio_path: str, output_path: str):
    """Replace audio track in video using ffmpeg."""
    # Use ffmpeg to replace audio (fast, no re-encoding video)
    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
        '-i', video_path,
        '-i', new_audio_path,
        '-c:v', 'copy',  # Copy video stream (no re-encode)
        '-c:a', 'aac',   # Encode audio to AAC
        '-map', '0:v:0',  # Use video from first input
        '-map', '1:a:0',  # Use audio from second input
        '-shortest',      # End when shortest stream ends
        output_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True, None
    except subprocess.CalledProcessError as e:
        return False, f"FFmpeg error: {e.stderr}"


def loop_video_to_match_audio(video_path: str, audio_duration: float, output_path: str):
    """Loop video to match audio duration."""
    # Get video duration
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = frame_count / fps if fps > 0 else 0
    cap.release()
    
    if video_duration >= audio_duration:
        # Video is long enough, just trim audio
        return video_path
    
    # Calculate how many loops needed
    loops_needed = int(np.ceil(audio_duration / video_duration))
    
    # Use ffmpeg to loop video
    temp_list = os.path.join(CACHE_DIR, "video_list.txt")
    with open(temp_list, 'w') as f:
        for _ in range(loops_needed):
            f.write(f"file '{os.path.abspath(video_path)}'\n")
    
    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
        '-f', 'concat', '-safe', '0',
        '-i', temp_list,
        '-c', 'copy',
        output_path
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        os.remove(temp_list)
        return output_path
    except:
        if os.path.exists(temp_list):
            os.remove(temp_list)
        return video_path  # Fallback to original


def generate_video_cunning(text: str, use_cached_voice: bool = False):
    """ULTRA-FAST generation: Only replace audio in template video."""
    import time
    start_time = time.time()
    
    # Check template video exists
    if not os.path.exists(TEMPLATE_VIDEO_FILE):
        return None, "❌ Template video not found. Run 'Generate Template Video' in Setup tab first."
    
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
    
    # Get audio duration
    audio_seg = AudioSegment.from_wav(audio_path)
    audio_duration = len(audio_seg) / 1000.0  # seconds
    
    # Prepare video (loop if needed to match audio length)
    video_to_use = TEMPLATE_VIDEO_FILE
    if audio_duration > 0:
        looped_video = os.path.join(CACHE_DIR, f"looped_template_{ts}.mp4")
        video_to_use = loop_video_to_match_audio(TEMPLATE_VIDEO_FILE, audio_duration, looped_video)
    
    # Replace audio track
    print("🔄 Replacing audio track in template video...")
    gen_dir = os.path.join(RESULT_DIR, f"gen_{ts}")
    os.makedirs(gen_dir, exist_ok=True)
    final_video = os.path.join(gen_dir, f"result_{ts}.mp4")
    
    success, error = replace_audio_in_video(video_to_use, audio_path, final_video)
    
    if not success:
        return None, f"❌ Failed to replace audio: {error}"
    
    elapsed = time.time() - start_time
    speedup = 87 / elapsed if elapsed > 0 else 1
    print(f"✅ Complete! ({elapsed:.1f}s, ~{speedup:.1f}x faster than optimized baseline)")
    
    return final_video, f"✅ Generated in {elapsed:.1f}s (~{speedup:.1f}x faster)\n📹 {os.path.basename(final_video)}\n🎭 Using template video + new audio"


def auto_setup_from_assets(size: int = 256):
    """Automatically setup using default assets."""
    face_cache = load_face_cache()
    voice_cache = load_voice_cache()
    
    results = []
    errors = []
    
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
        errors.append(f"⚠ Image not found: {DEFAULT_IMAGE}")
    
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
        errors.append(f"⚠ Voice not found")
    
    output = "\n".join(results) if results else ""
    if errors:
        output += "\n\n" + "\n".join(errors)
    
    return output if output else "✅ Ready! Face and voice cached."


# Gradio UI
with gr.Blocks(title="SadTalker — Cunning (Template Video)") as demo:
    gr.Markdown(f"""
    # 🎭 SadTalker: Cunning Mode (Ultra-Fast Template Video)
    
    **Cunning optimization:** Pre-render template video once → Replace audio for new text
    
    **Expected speed:** ~1-3s per generation (vs ~87s optimized, ~360s standard)
    
    **Trade-off:** Lip movements use template (may not perfectly match new text)
    
    **📁 Cache:** `{CACHE_DIR}`  
    **📁 Results:** `{RESULT_DIR}`  
    **📁 Assets:** `{ASSETS_DIR}`
    """)
    
    with gr.Tabs():
        with gr.TabItem("1️⃣ Setup (Run Once)"):
            gr.Markdown("### Step 1: Pre-process face + voice")
            
            with gr.Row():
                with gr.Column():
                    setup_size = gr.Slider(
                        minimum=128, maximum=512, step=64, value=256,
                        label="Face Resolution (px)"
                    )
                    auto_setup_btn = gr.Button("🚀 Auto-Setup from Assets", variant="primary")
                auto_setup_status = gr.Textbox(label="Setup Status", interactive=False, lines=5)
            
            auto_setup_btn.click(fn=auto_setup_from_assets, inputs=[setup_size], outputs=[auto_setup_status])
            
            gr.Markdown("---\n### Step 2: Generate Template Video (Once)")
            gr.Markdown("**Generate a template video with lip movements. This will be reused for all future generations.**")
            
            with gr.Row():
                with gr.Column():
                    template_text = gr.Textbox(
                        label="Template Text (Optional)",
                        placeholder="Enter text for template video (covers common phonemes). Leave empty for default.",
                        lines=3
                    )
                    template_audio = gr.Audio(
                        type="filepath",
                        label="Or Upload Template Audio (Optional)"
                    )
                    gen_template_btn = gr.Button("🎭 Generate Template Video", variant="primary", scale=2)
                
                template_status = gr.Textbox(label="Template Status", interactive=False, lines=5)
            
            def do_generate_template(text, audio, size):
                audio_path = None
                if audio:
                    audio_path = audio if isinstance(audio, str) else audio.get("path") or getattr(audio, "name", None)
                video_path, msg = generate_template_video(
                    template_text=text if text and text.strip() else None,
                    template_audio_path=audio_path,
                    size=int(size)
                )
                return msg
            
            gen_template_btn.click(
                fn=do_generate_template,
                inputs=[template_text, template_audio, setup_size],
                outputs=[template_status]
            )
            
            gr.Markdown("---\n### Manual Setup (Optional)")
            
            with gr.Row():
                with gr.Column():
                    setup_image = gr.Image(type="filepath", label="Face Image")
                    setup_cache_id = gr.Textbox(label="Face Cache ID", value="default")
                    setup_face_size = gr.Slider(minimum=128, maximum=512, step=64, value=256, label="Resolution")
                    setup_face_btn = gr.Button("Pre-process Face", variant="secondary")
                
                with gr.Column():
                    setup_voice = gr.Audio(type="filepath", label="Voice Audio File")
                    setup_voice_id = gr.Textbox(label="Voice Cache ID", value="default")
                    setup_voice_btn = gr.Button("Cache Voice", variant="secondary")
            
            setup_status = gr.Textbox(label="Manual Setup Status", interactive=False, lines=3)
            
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
            gr.Markdown("### Enter text → Instant video (audio replacement only)")
            
            gen_text = gr.Textbox(
                label="Text to speak",
                lines=4,
                placeholder="Enter the text for the avatar to read..."
            )
            
            with gr.Row():
                gen_mode = gr.Radio(
                    choices=["Use TTS (Text-to-Speech)", "Use Cached Voice File"],
                    value="Use TTS (Text-to-Speech)",
                    label="Audio Source"
                )
                gen_btn = gr.Button("⚡ Generate Video (Ultra-Fast)", variant="primary", scale=2)
            
            gen_video = gr.Video(label="Output Video", height=400)
            gen_status = gr.Textbox(label="Status", interactive=False, lines=4)
            
            def do_generate(text, mode):
                if not text or not text.strip():
                    return None, "Please enter some text"
                
                use_cached = (mode == "Use Cached Voice File")
                video_path, status = generate_video_cunning(text, use_cached_voice=use_cached)
                return video_path, status
            
            gen_btn.click(fn=do_generate, inputs=[gen_text, gen_mode], outputs=[gen_video, gen_status])
    
    gr.Markdown("""
    ### 🎭 How Cunning Mode Works:
    1. **Setup:** Pre-process face + Generate template video with lip movements (once, ~87s)
    2. **Generate:** Text → TTS → Replace audio in template video (~1-3s)
    3. **Speed:** ~30-90x faster than standard generation!
    
    ### ⚡ Performance:
    - **Standard:** ~360s
    - **Optimized:** ~87s
    - **Faster:** ~50-60s
    - **Cunning:** ~1-3s ⚡⚡⚡
    
    ### 💡 Tips:
    - Use phoneme-rich template text (covers common mouth movements)
    - Template video loops automatically if new audio is longer
    - Lip sync may not be perfect (uses template movements)
    """)

if __name__ == "__main__":
    print("\n🎭 Starting SadTalker Cunning App (Template Video Mode)...")
    print(f"📁 Working directory: {BASE_DIR}")
    print(f"💻 CUDA available: {torch.cuda.is_available()}\n")
    
    demo.launch(
        debug=True,
        share=False,
        server_name="127.0.0.1",
        server_port=7862  # Different port
    )
