@echo off
REM Install SadTalker Requirements for Windows
REM Make sure you're in your venv before running this

echo Installing PyTorch (CPU version - change to CUDA if you have GPU)...
pip install torch torchvision torchaudio

echo.
echo Installing SadTalker requirements...
pip install -r SadTalker\requirements.txt

echo.
echo Installation complete!
pause
