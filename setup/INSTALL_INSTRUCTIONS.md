# Installing SadTalker Requirements

## Quick Install (Windows)

**Make sure your virtual environment is activated first!**

```bash
# Activate venv if not already active
venv\Scripts\activate

# Install PyTorch (CPU version - recommended for most users)
pip install torch torchvision torchaudio

# OR if you have CUDA GPU, install CUDA version:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install SadTalker requirements
pip install -r SadTalker\requirements.txt

# Install Gradio UI dependencies (if not already installed)
pip install gradio moviepy pydub
```

## Step-by-Step

1. **Activate your virtual environment:**
   ```bash
   cd D:\WorkDocs\Projects\FlaskProjects\sadTalker20260209\app\SadTalker
   venv\Scripts\activate
   ```

2. **Install PyTorch:**
   ```bash
   # CPU version (works on all systems)
   pip install torch torchvision torchaudio
   
   # OR CUDA version (if you have NVIDIA GPU with CUDA 11.8)
   # pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Install SadTalker requirements:**
   ```bash
   cd ..
   pip install -r SadTalker\requirements.txt
   ```

4. **Note:** You may encounter the `scikit-image` build error again. If so:
   ```bash
   pip install --only-binary=scikit-image scikit-image==0.19.3
   ```

## Verify Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import gradio; print('Gradio installed')"
```

## Troubleshooting

- **If scikit-image fails:** Use pre-built wheel: `pip install --only-binary=scikit-image scikit-image==0.19.3`
- **If dlib fails:** Try `pip install dlib-bin` (Windows) or `pip install dlib` (Linux/Mac)
- **If ffmpeg missing:** Install from https://ffmpeg.org/download.html or use `scoop install ffmpeg`
