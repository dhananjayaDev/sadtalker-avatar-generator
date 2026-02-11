# SadTalker Local Setup Guide

## Quick Setup Steps

### 1. Clone SadTalker Repository
```bash
git clone https://github.com/OpenTalker/SadTalker.git
cd SadTalker
```

### 2. Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
pip install gradio moviepy pydub
```

### 4. Download Checkpoints
```bash
# Windows (PowerShell)
bash scripts/download_models.sh

# Or manually download models if bash doesn't work
```

### 5. Run the Gradio UI
```bash
python sadtalker_local.py
```

The UI will open at `http://127.0.0.1:7860`

## File Structure

```
SadTalker/
├── inference.py
├── requirements.txt
├── scripts/
├── checkpoints/
├── results/          # Generated videos saved here
└── sadtalker_local.py  # Copy this file here
```

## Notes

- **Virtual Environment**: Highly recommended to avoid dependency conflicts
- **Python Version**: Python 3.8+ required
- **GPU**: CUDA GPU recommended for faster processing (CPU works but slower)
- **First Run**: May take longer due to model loading

## Troubleshooting

### Issue: Module not found
- Activate your virtual environment
- Run `pip install -r requirements.txt` again

### Issue: inference.py not found
- Make sure you're running `sadtalker_local.py` from the SadTalker directory
- Or edit `BASE_DIR` in the script to point to your SadTalker path

### Issue: CUDA errors
- Check if you have CUDA installed
- SadTalker will fall back to CPU if GPU not available
