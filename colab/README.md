# Running SadTalker in Google Colab (from your GitHub repo)

This folder contains notebooks you can run in [Google Colab](https://colab.research.google.com) using your repo: **https://github.com/dhananjayaDev/sadtalker-avatar-generator**

---

## Quick start: one-click notebook

1. **Push this project to your GitHub** (if not already):
   - Repo: `https://github.com/dhananjayaDev/sadtalker-avatar-generator`
   - Make sure the `colab` folder and this notebook are in the repo.

2. **Open the notebook in Colab** (choose one):
   - **From Colab:** **File → Open notebook → GitHub** tab → paste:
     ```text
     https://github.com/dhananjayaDev/sadtalker-avatar-generator
     ```
     Then open `colab/run_from_github.ipynb`.
   - **Direct link:**
     ```text
     https://colab.research.google.com/github/dhananjayaDev/sadtalker-avatar-generator/blob/main/colab/run_from_github.ipynb
     ```

3. **Enable GPU:** **Runtime → Change runtime type → Hardware accelerator → GPU** (e.g. T4).

4. **Run all cells** in order. The notebook will:
   - Download your repo
   - Install dependencies
   - Download checkpoints
   - Launch the Gradio app (with a public link if you use `--share`).

---

## What "linking" the repo means

- **Opening a notebook from GitHub:** Colab loads that notebook from your repo. The runtime does **not** automatically have the rest of the repo; the notebook downloads the repo (ZIP) in Step 2.
- **Step 2:** The notebook downloads the repo (ZIP) and changes into it so all later steps (install, download models, run app) use your project.

---

## Notebooks in this folder

| Notebook | Description |
|----------|-------------|
| **run_from_github.ipynb** | Downloads your GitHub repo and runs SadTalker (Gradio or inference). Use this to "link" Colab to your repo. |
| quick_demo.ipynb | Original SadTalker quick demo (may point to another repo). |
| colab_minimal_setup.ipynb | Minimal setup without full clone. |
| colab_optimized_cached.ipynb | Cached face/voice setup for faster runs. |
| colab_single_image_voice_text.ipynb | Single image + TTS (edge-tts) + text UI. |

For running **your** fork with minimal setup, use **run_from_github.ipynb**.
