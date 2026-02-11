"""
Download SadTalker main model checkpoints (Windows-friendly).
Run from SadTalker directory: python download_sadtalker_checkpoints.py

This downloads the safetensor models so the app uses the new version
and does not require the old .pth files (epoch_20.pth, etc.).
"""
import os
import sys
import time
import urllib.request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINTS_DIR = os.path.join(SCRIPT_DIR, "checkpoints")
BASE_URL = "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc"

DOWNLOADS = [
    ("mapping_00109-model.pth.tar", "mapping_00109-model.pth.tar"),
    ("mapping_00229-model.pth.tar", "mapping_00229-model.pth.tar"),
    ("SadTalker_V0.0.2_256.safetensors", "SadTalker_V0.0.2_256.safetensors"),
    ("SadTalker_V0.0.2_512.safetensors", "SadTalker_V0.0.2_512.safetensors"),
]

def download_with_retries(url, dest_path, max_retries=5):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
    for attempt in range(1, max_retries + 1):
        try:
            print(f"  Attempt {attempt}/{max_retries}...")
            with urllib.request.urlopen(req, timeout=120) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                with open(dest_path, "wb") as f:
                    downloaded = 0
                    chunk = 1024 * 1024
                    while True:
                        data = resp.read(chunk)
                        if not data:
                            break
                        f.write(data)
                        downloaded += len(data)
                        if total:
                            pct = 100 * downloaded / total
                            print(f"\r  {downloaded // (1024*1024)} MB ({pct:.1f}%)", end="", flush=True)
            print()
            return True
        except Exception as e:
            print(f"  Error: {e}")
            if attempt < max_retries:
                wait = 10 * attempt
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
    return False

def main():
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    print(f"Downloading SadTalker checkpoints to: {CHECKPOINTS_DIR}\n")

    for filename, dest_name in DOWNLOADS:
        dest = os.path.join(CHECKPOINTS_DIR, dest_name)
        if os.path.exists(dest):
            print(f"[SKIP] {dest_name} (already exists)")
            continue
        url = f"{BASE_URL}/{filename}"
        print(f"Downloading {dest_name}...")
        if download_with_retries(url, dest):
            print(f"  OK: {dest_name}\n")
        else:
            print(f"  FAILED: {dest_name}")
            sys.exit(1)

    print("All SadTalker checkpoints downloaded. The app will use the new safetensor models.")

if __name__ == "__main__":
    main()
