# Fixing scikit-image Build Error

## Problem
The `scikit-image` package failed to build because it requires `clang-cl.exe` (LLVM/Clang compiler) which is not installed on your system.

## Solutions

### Option 1: Install Pre-built Wheel (Recommended - Easiest)
Skip building from source and use a pre-built wheel:

```bash
# In your venv
pip install --only-binary=scikit-image scikit-image
```

If that doesn't work, try installing a specific version:
```bash
pip install scikit-image==0.21.0
```

### Option 2: Install LLVM/Clang (If you need to build from source)
1. Download LLVM from: https://github.com/llvm/llvm-project/releases
2. Or install via Visual Studio Installer:
   - Open Visual Studio Installer
   - Modify your installation
   - Add "C++ Clang Compiler for Windows" component

### Option 3: Skip scikit-image (If not needed)
If SadTalker doesn't strictly require scikit-image, you can skip it:

```bash
pip install -r requirements.txt --ignore-installed scikit-image
```

Then install only what's needed:
```bash
pip install gradio moviepy pydub
```

### Option 4: Use Conda (Alternative)
If pip continues to fail, use conda which has pre-built binaries:

```bash
conda install -c conda-forge scikit-image
```

## Quick Fix Command
Try this first:
```bash
pip install --upgrade pip
pip install --only-binary=scikit-image scikit-image
pip install gradio moviepy pydub
```

## Note
The `scikit-image` build error won't prevent you from running the Gradio UI if SadTalker's core dependencies are installed. The file location error is the main issue preventing execution.
