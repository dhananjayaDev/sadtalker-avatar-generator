"""
SadTalker Gradio UI for Google Colab
This script creates a Gradio interface for generating talking avatar videos
with real-time progress updates and log streaming.
"""

# ============================================================================
# SETUP CELLS FOR COLAB (Run these first)
# ============================================================================
"""
# Cell 1: Clone and Setup SadTalker
!git clone https://github.com/OpenTalker/SadTalker.git
%cd SadTalker

# Cell 2: Install Dependencies
!pip install -r requirements.txt
!pip install gradio moviepy pydub

# Cell 3: Download Checkpoints
!bash scripts/download_models.sh
"""

# ============================================================================
# MAIN APPLICATION CODE
# ============================================================================

import gradio as gr
import subprocess
import time
import os
import threading
import queue
from pathlib import Path
from datetime import datetime

BASE_DIR = "/content/SadTalker"
RESULT_DIR = f"{BASE_DIR}/results"
os.makedirs(RESULT_DIR, exist_ok=True)

# Global variables for progress tracking
current_process = None
log_queue = queue.Queue()
process_start_time = None


def run_inference_with_streaming(image_path, audio_path, progress=gr.Progress()):
    """
    Run SadTalker inference with real-time log streaming
    """
    global current_process, process_start_time
    
    if image_path is None or audio_path is None:
        return "❌ Please upload both image and audio files", "", None, None
    
    # Validate files exist
    if not os.path.exists(image_path):
        return "❌ Image file not found", "", None, None
    if not os.path.exists(audio_path):
        return "❌ Audio file not found", "", None, None
    
    process_start_time = time.time()
    start_time_str = datetime.now().strftime("%H:%M:%S")
    
    # Clear log queue
    while not log_queue.empty():
        try:
            log_queue.get_nowait()
        except:
            pass
    
    # Build command
    cmd = [
        "python", "inference.py",
        "--driven_audio", audio_path,
        "--source_image", image_path,
        "--result_dir", RESULT_DIR,
        "--enhancer", "gfpgan",
        "--still",
        "--preprocess", "full"
    ]
    
    logs = []
    status_msg = f"⏳ Started at {start_time_str}\n"
    
    try:
        # Start process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        current_process = process
        
        # Stream output in real-time
        for line in iter(process.stdout.readline, ''):
            if line:
                line = line.strip()
                logs.append(line)
                # Update progress with latest log
                if progress is not None:
                    progress(0, desc=f"Processing... {line[:50]}")
                yield (
                    f"⏳ Processing... ({len(logs)} lines logged)",
                    "\n".join(logs[-20:]),  # Show last 20 lines
                    None,
                    None
                )
        
        process.wait()
        
        # Calculate elapsed time
        elapsed = time.time() - process_start_time
        elapsed_str = f"{elapsed:.2f} seconds ({elapsed/60:.2f} minutes)"
        
        # Find latest video
        output_video = None
        videos = sorted(Path(RESULT_DIR).rglob("*.mp4"), key=os.path.getmtime, reverse=True)
        if videos:
            output_video = str(videos[0])
            status_msg = f"✅ Generation completed!\n⏱️ Time taken: {elapsed_str}\n📹 Output: {os.path.basename(output_video)}"
        else:
            status_msg = f"⚠️ Process completed but no video found in {RESULT_DIR}"
        
        yield (
            status_msg,
            "\n".join(logs[-30:]),  # Show last 30 lines
            output_video,
            elapsed_str
        )
        
    except Exception as e:
        error_msg = f"❌ Error occurred: {str(e)}"
        yield (
            error_msg,
            "\n".join(logs) + f"\n\nERROR: {str(e)}",
            None,
            None
        )
    finally:
        current_process = None


def generate_video(image, audio, progress=gr.Progress()):
    """
    Main function called by Gradio
    """
    if image is None or audio is None:
        return "❌ Please upload both image and audio files", "", None, None
    
    # Handle Gradio file paths
    image_path = image if isinstance(image, str) else image.name if hasattr(image, 'name') else image
    audio_path = audio if isinstance(audio, str) else audio if isinstance(audio, dict) else None
    
    # Handle audio dict format from Gradio
    if isinstance(audio_path, dict):
        audio_path = audio_path.get('name', audio_path.get('path', None))
    
    if not image_path or not audio_path:
        return "❌ Invalid file paths", "", None, None
    
    # Run inference with streaming
    for status, logs, video, elapsed in run_inference_with_streaming(image_path, audio_path, progress):
        yield status, logs, video, elapsed


def cancel_generation():
    """Cancel the current generation process"""
    global current_process
    if current_process:
        current_process.terminate()
        current_process = None
        return "🛑 Generation cancelled"
    return "ℹ️ No process running"


# Create Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎭 SadTalker Avatar Generator (Colab)
    
    Upload **one face image** and **one audio file** to generate a talking avatar video.
    
    **Features:**
    - Real-time generation progress
    - Live log streaming
    - Automatic video output display
    - Time tracking
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            image = gr.Image(
                type="filepath",
                label="📷 Upload Face Image",
                height=300
            )
            audio = gr.Audio(
                type="filepath",
                label="🎵 Upload Audio File",
                sources=["upload", "microphone"]
            )
            
            with gr.Row():
                run_btn = gr.Button(
                    "🚀 Generate Avatar Video",
                    variant="primary",
                    size="lg"
                )
                cancel_btn = gr.Button(
                    "🛑 Cancel",
                    variant="stop",
                    size="lg"
                )
    
    with gr.Row():
        with gr.Column():
            status = gr.Textbox(
                label="📊 Status",
                value="Ready to generate...",
                interactive=False
            )
            
            elapsed_time = gr.Textbox(
                label="⏱️ Elapsed Time",
                value="",
                interactive=False
            )
    
    with gr.Row():
        logs = gr.Textbox(
            label="📝 Generation Logs (Real-time)",
            lines=15,
            max_lines=30,
            interactive=False,
            show_copy_button=True
        )
    
    with gr.Row():
        video = gr.Video(
            label="🎬 Final Output Video",
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
    
    # Examples (optional - add your own example files)
    gr.Markdown("### 💡 Tips:")
    gr.Markdown("""
    - Use a clear face image (front-facing works best)
    - Audio should be clear and not too long for faster processing
    - First generation may take longer due to model loading
    - Check the logs for detailed progress information
    """)

# Launch the interface
if __name__ == "__main__":
    demo.launch(
        debug=True,
        share=True,  # Creates a public link
        server_name="0.0.0.0",
        server_port=7860
    )
