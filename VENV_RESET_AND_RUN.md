# Reset venv and run Flask app (Windows PowerShell)

After a PC reset, use these commands from the **SadTalker** folder.

---

## 1. Go to project folder

```powershell
cd "d:\WorkDocs\Projects\FlaskProjects\sadTalker20260209\app\SadTalker"
```

---

## 2. Remove existing venv (if any)

Use the name you actually have (`venv`, `.venv`, or `env`):

```powershell
# If your folder is named "venv":
Remove-Item -Recurse -Force venv -ErrorAction SilentlyContinue

# Or if it's ".venv":
Remove-Item -Recurse -Force .venv -ErrorAction SilentlyContinue

# Or "env":
Remove-Item -Recurse -Force env -ErrorAction SilentlyContinue
```

---

## 3. Create a new virtual environment

```powershell
python -m venv venv
```

---

## 4. Activate the venv

```powershell
.\venv\Scripts\Activate.ps1
```

If you get an execution policy error, run once (as Administrator if needed):

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then run the activate command again.

---

## 5. Upgrade pip and install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install flask
```

*(Flask is required for `sample_API.py` but not in requirements.txt.)*

---

## 6. Run the Flask app

```powershell
python sample_API.py
```

API will be at **http://0.0.0.0:8000** (or http://localhost:8000).  
Health check: http://localhost:8000/health

---

## Optional: run video.py Gradio UI instead

```powershell
python video.py --ui
```

---

## Use GPU (faster Setup — avoid hours on CPU)

The app already uses the GPU when PyTorch can see CUDA. If Setup takes hours, PyTorch is likely **CPU-only**.

**1. Check (with venv activated):**

```powershell
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

If it prints **`CUDA available: False`**, install the CUDA build of PyTorch.

**2. Install PyTorch with CUDA 12 (Windows):**

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Then run the check again. If it prints **`CUDA available: True`**, run Setup again; it should be much faster.

*(Your driver supports CUDA 12.7; the cu124 wheel is compatible.)*

---

## One-shot copy-paste (all steps)

Run from PowerShell in the SadTalker folder:

```powershell
cd "d:\WorkDocs\Projects\FlaskProjects\sadTalker20260209\app\SadTalker"
Remove-Item -Recurse -Force venv -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force .venv -ErrorAction SilentlyContinue
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install flask
python sample_API.py
```

After the first time, you only need:

```powershell
cd "d:\WorkDocs\Projects\FlaskProjects\sadTalker20260209\app\SadTalker"
.\venv\Scripts\Activate.ps1
python sample_API.py
```
