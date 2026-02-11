# SadTalker Gradio UI for Google Colab

This repository contains ready-to-use code for running SadTalker with a Gradio UI in Google Colab.

## 📁 Files

- **`sadtalker_colab_simple.py`** - Single-file version (easiest to use)
- **`sadtalker_gradio_colab.py`** - Full-featured version with detailed comments
- **`sadtalker_colab_notebook.ipynb`** - Jupyter notebook format

## 🚀 Quick Start (Google Colab)

### Option 1: Using the Simple Python File

1. **Open a new Colab notebook**

2. **Run Setup Cells:**

```python
# Cell 1: Clone SadTalker
!git clone https://github.com/OpenTalker/SadTalker.git
%cd SadTalker

# Cell 2: Install dependencies
!pip install -r requirements.txt
!pip install gradio moviepy pydub

# Cell 3: Download checkpoints
!bash scripts/download_models.sh
```

3. **Copy and paste the entire `sadtalker_colab_simple.py` file into a new cell and run it**

### Option 2: Using the Notebook

1. Upload `sadtalker_colab_notebook.ipynb` to Google Colab
2. Run all cells sequentially

## ✨ Features

- ✅ **Real-time Progress Updates** - See generation progress as it happens
- ✅ **Live Log Streaming** - Watch detailed logs update in real-time
- ✅ **Time Tracking** - Monitor elapsed time during generation
- ✅ **Video Preview** - Automatically displays the generated video
- ✅ **Cancel Support** - Stop generation if needed
- ✅ **Error Handling** - Clear error messages if something goes wrong

## 🎯 Usage

1. **Upload Image**: Click on the image upload area and select a face image
2. **Upload Audio**: Click on the audio upload area and select an audio file
3. **Generate**: Click the "🚀 Generate Video" button
4. **Monitor**: Watch the status, logs, and elapsed time update in real-time
5. **View Result**: The generated video will appear automatically when complete

## 📝 Notes

- **First Run**: The first generation may take longer due to model loading
- **Image Requirements**: Use clear, front-facing face images for best results
- **Audio**: Shorter audio files process faster
- **Public Link**: The UI creates a public Gradio link that you can share
- **Storage**: Generated videos are saved in `/content/SadTalker/results/`

## 🔧 Troubleshooting

### Issue: "No video found"
- Check the logs for error messages
- Ensure both image and audio files are valid
- Verify that the inference completed successfully

### Issue: Process hangs
- Use the "Cancel" button to stop
- Check Colab runtime (may need to restart)
- Ensure sufficient GPU/RAM is allocated

### Issue: Import errors
- Make sure all setup cells ran successfully
- Try restarting runtime and running setup again
- Check that you're in the correct directory (`/content/SadTalker`)

## 📊 UI Components

- **Status Box**: Shows current generation status
- **Elapsed Time**: Real-time timer during generation
- **Logs Box**: Detailed generation logs (scrollable, copyable)
- **Video Player**: Displays the final generated video

## 🎨 Customization

You can modify the code to:
- Change output directory
- Adjust inference parameters (enhancer, preprocessing options)
- Customize UI theme
- Add example files
- Modify video quality settings

## 📚 SadTalker Parameters

The code uses these inference parameters:
- `--enhancer gfpgan`: Face enhancement
- `--still`: Keep face still (no head movement)
- `--preprocess full`: Full preprocessing pipeline

You can modify these in the `cmd` list within the `generate_video` function.

## 🔗 Links

- [SadTalker GitHub](https://github.com/OpenTalker/SadTalker)
- [Gradio Documentation](https://www.gradio.app/docs/)
