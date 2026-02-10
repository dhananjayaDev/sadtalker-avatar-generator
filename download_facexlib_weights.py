"""
Download facexlib and GFPGAN weights for SadTalker (Windows-friendly).
Run from SadTalker directory: python download_facexlib_weights.py
"""
import os
import sys
import time
import urllib.request

# Run from SadTalker directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(SCRIPT_DIR, "gfpgan", "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# Same URLs as in download_models.sh (facexlib + GFPGAN)
DOWNLOADS = [
    ("https://github.com/xinntao/facexlib/releases/download/v0.1.0/alignment_WFLW_4HG.pth", "alignment_WFLW_4HG.pth"),
    ("https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth", "detection_Resnet50_Final.pth"),
    ("https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth", "GFPGANv1.4.pth"),
    ("https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth", "parsing_parsenet.pth"),
]

def download_with_retries(url, dest_path, max_retries=5):
    # Use a browser User-Agent to avoid GitHub 403/500 issues
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
    for attempt in range(1, max_retries + 1):
        try:
            print(f"  Attempt {attempt}/{max_retries}: {url}")
            with urllib.request.urlopen(req, timeout=60) as resp:
                with open(dest_path, "wb") as f:
                    f.write(resp.read())
            return True
        except Exception as e:
            print(f"  Error: {e}")
            if attempt < max_retries:
                wait = 10 * attempt
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
    return False

def main():
    print(f"Downloading weights to: {WEIGHTS_DIR}\n")
    for url, filename in DOWNLOADS:
        dest = os.path.join(WEIGHTS_DIR, filename)
        if os.path.exists(dest):
            print(f"[SKIP] {filename} (already exists)")
            continue
        print(f"Downloading {filename}...")
        if download_with_retries(url, dest):
            print(f"  OK: {filename}\n")
        else:
            print(f"  FAILED: {filename}")
            print("  You can try again later (GitHub may return 500 temporarily).")
            print("  Or download manually and place in:", WEIGHTS_DIR)
            sys.exit(1)
    print("All weights downloaded successfully.")

if __name__ == "__main__":
    main()
