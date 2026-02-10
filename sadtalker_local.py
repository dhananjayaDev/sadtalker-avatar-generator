"""
SadTalker Gradio UI - Local Version
Run this script locally after setting up SadTalker
"""

import gradio as gr
import subprocess
import time
import os
import sys
from pathlib import Path
from datetime import datetime

# Configuration - automatically find SadTalker directory
# This script can be run from the app directory or SadTalker directory
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()

# Try multiple locations to find SadTalker
possible_paths = [
    script_dir,  # If running from SadTalker directory
    os.path.join(script_dir, "SadTalker"),  # If running from app directory (SadTalker subfolder)
    os.path.dirname(script_dir),  # Parent directory
]

BASE_DIR = None
for path in possible_paths:
    if os.path.exists(os.path.join(path, "inference.py")):
        BASE_DIR = path
        break

if BASE_DIR is None:
    print(f"⚠️ Error: Could not find SadTalker directory!")
    print(f"Searched in: {possible_paths}")
    print("\nPlease ensure:")
    print("1. SadTalker folder exists with inference.py")
    print("2. Or set BASE_DIR manually in the script")
    sys.exit(1)

RESULT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

print(f"📁 Using BASE_DIR: {BASE_DIR}")
print(f"📁 Results will be saved to: {RESULT_DIR}")

current_process = None
process_start_time = None


def generate_video(image, audio, progress=gr.Progress()):
    """Generate video with real-time progress updates"""
    global current_process, process_start_time
    
    # Validation
    if image is None or audio is None:
        return "❌ Please upload both image and audio files", "", None, None
    
    # Handle file paths (Gradio may return different formats)
    image_path = image if isinstance(image, str) else (image.name if hasattr(image, 'name') else image)
    audio_path = audio if isinstance(audio, str) else (audio if isinstance(audio, dict) else None)
    
    if isinstance(audio_path, dict):
        audio_path = audio_path.get('name') or audio_path.get('path')
    
    if not image_path or not audio_path:
        return "❌ Invalid file paths", "", None, None
    
    if not os.path.exists(image_path):
        return "❌ Image file not found", "", None, None
    if not os.path.exists(audio_path):
        return "❌ Audio file not found", "", None, None
    
    # Start timing
    process_start_time = time.time()
    start_time_str = datetime.now().strftime("%H:%M:%S")
    
    # Build command - use absolute path to inference.py
    inference_script = os.path.join(BASE_DIR, "inference.py")
    if not os.path.exists(inference_script):
        return f"❌ inference.py not found at {inference_script}", "", None, None
    
    # Use sys.executable to ensure we use the correct Python interpreter
    cmd = [
        sys.executable, inference_script,
        "--driven_audio", audio_path,
        "--source_image", image_path,
        "--result_dir", RESULT_DIR,
        "--enhancer", "gfpgan",
        "--still",
        "--preprocess", "full"
    ]
    
    logs = []
    status_msg = f"⏳ Started at {start_time_str}"
    
    try:
        # Start process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            cwd=BASE_DIR  # Run from SadTalker directory
        )
        current_process = process
        
        # Stream output line by line
        for line in iter(process.stdout.readline, ''):
            if line:
                line = line.strip()
                if line:  # Skip empty lines
                    logs.append(line)
                    elapsed = time.time() - process_start_time
                    
                    # Update progress bar
                    if progress is not None:
                        progress(0, desc=f"Processing... {line[:50]}")
                    
                    # Yield updates (last 25 lines for display)
                    yield (
                        f"⏳ Processing... ({len(logs)} lines) | ⏱️ {elapsed:.1f}s",
                        "\n".join(logs[-25:]),
                        None,
                        f"{elapsed:.2f} seconds"
                    )
        
        # Wait for completion
        process.wait()
        
        # Calculate final time
        elapsed = time.time() - process_start_time
        elapsed_str = f"{elapsed:.2f} seconds ({elapsed/60:.2f} minutes)"
        
        # Find output video
        output_video = None
        videos = sorted(Path(RESULT_DIR).rglob("*.mp4"), key=os.path.getmtime, reverse=True)
        
        if videos:
            output_video = str(videos[0])
            status_msg = f"✅ Completed!\n⏱️ Time: {elapsed_str}\n📹 Video: {os.path.basename(output_video)}"
        else:
            status_msg = f"⚠️ Completed but no video found. Check logs for errors."
        
        # Final yield with video
        yield (
            status_msg,
            "\n".join(logs[-35:]),  # Show last 35 lines
            output_video,
            elapsed_str
        )
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        yield (
            error_msg,
            "\n".join(logs) + f"\n\n❌ ERROR: {str(e)}",
            None,
            None
        )
    finally:
        current_process = None


def cancel_generation():
    """Cancel current generation"""
    global current_process
    if current_process:
        current_process.terminate()
        current_process = None
        return "🛑 Cancelled"
    return "ℹ️ No active process"


# Create Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown(f"""
    # 🎭 SadTalker Avatar Generator (Local)
    
    Upload a **face image** and **audio file** to create a talking avatar video.
    
    **✨ Features:**
    - Real-time progress updates
    - Live generation logs
    - Automatic video preview
    - Time tracking
    
    **📁 Results saved to:** `{RESULT_DIR}`
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            image = gr.Image(
                type="filepath",
                label="📷 Face Image",
                height=300
            )
            audio = gr.Audio(
                type="filepath",
                label="🎵 Audio File",
                sources=["upload"]
            )
            
            with gr.Row():
                run_btn = gr.Button(
                    "🚀 Generate Video",
                    variant="primary",
                    size="lg"
                )
                cancel_btn = gr.Button(
                    "🛑 Cancel",
                    variant="stop"
                )
    
    with gr.Row():
        status = gr.Textbox(
            label="📊 Status",
            value="Ready...",
            interactive=False
        )
        elapsed_time = gr.Textbox(
            label="⏱️ Elapsed Time",
            value="",
            interactive=False
        )
    
    logs = gr.Textbox(
        label="📝 Generation Logs (Real-time)",
        lines=12,
        max_lines=30,
        interactive=False
    )
    
    video = gr.Video(
        label="🎬 Output Video",
        height=400
    )
    
    # Event handlers
    run_btn.click(
        fn=generate_video,
        inputs=[image, audio],
        outputs=[status, logs, video, elapsed_time],
        show_progress="full"
    )
    
    cancel_btn.click(
        fn=cancel_generation,
        outputs=[status]
    )
    
    gr.Markdown("""
    ### 💡 Tips:
    - Use clear, front-facing face images
    - Shorter audio files process faster
    - First run may take longer (model loading)
    - Check logs for detailed progress
    """)

# Launch
if __name__ == "__main__":
    demo.launch(
        debug=True,
        share=False,  # Set to True if you want a public link
        server_name="127.0.0.1",  # Localhost
        server_port=7860,
        theme=gr.themes.Soft()  # Theme moved here for Gradio 6.0+
    )
