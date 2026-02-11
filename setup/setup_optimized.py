"""
Quick setup script: Pre-process face + voice from assets folder
Run this once to cache face and voice, then use sadtalker_optimized_local.py
"""

import os
import sys

# Add current directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from sadtalker_optimized_local import (
    DEFAULT_IMAGE, DEFAULT_VOICE_MP3, DEFAULT_VOICE_WAV,
    preprocess_and_cache_face, preprocess_and_cache_voice,
    load_face_cache, load_voice_cache
)

def main():
    print("🚀 SadTalker Optimized Setup\n")
    
    # Check assets
    print("Checking assets...")
    has_image = os.path.exists(DEFAULT_IMAGE)
    has_voice_mp3 = os.path.exists(DEFAULT_VOICE_MP3)
    has_voice_wav = os.path.exists(DEFAULT_VOICE_WAV)
    
    if has_image:
        print(f"✓ Found image: {DEFAULT_IMAGE}")
    else:
        print(f"⚠ Image not found: {DEFAULT_IMAGE}")
    
    if has_voice_wav:
        print(f"✓ Found voice: {DEFAULT_VOICE_WAV}")
    elif has_voice_mp3:
        print(f"✓ Found voice: {DEFAULT_VOICE_MP3}")
    else:
        print(f"⚠ Voice not found. Expected: {DEFAULT_VOICE_MP3}")
    
    print("\n" + "="*50)
    
    # Setup face
    face_cache = load_face_cache()
    if has_image:
        if not face_cache:
            print("\n🔄 Pre-processing face...")
            try:
                _, msg = preprocess_and_cache_face(DEFAULT_IMAGE, "female-01")
                print(msg)
            except Exception as e:
                print(f"❌ Face setup failed: {e}")
        else:
            print("\n✓ Face already cached")
    else:
        print("\n⚠ Skipping face setup (image not found)")
    
    # Setup voice
    voice_cache = load_voice_cache()
    voice_file = None
    if has_voice_wav:
        voice_file = DEFAULT_VOICE_WAV
    elif has_voice_mp3:
        voice_file = DEFAULT_VOICE_MP3
    
    if voice_file:
        if not voice_cache:
            print("\n🔄 Pre-processing voice...")
            try:
                _, msg = preprocess_and_cache_voice(voice_file, "female-01")
                print(msg)
            except Exception as e:
                print(f"❌ Voice setup failed: {e}")
        else:
            print("\n✓ Voice already cached")
    else:
        print("\n⚠ Skipping voice setup (voice file not found)")
    
    print("\n" + "="*50)
    print("\n✅ Setup complete!")
    print("\n📝 Next steps:")
    print("   1. Run: python sadtalker_optimized_local.py")
    print("   2. Open browser to: http://127.0.0.1:7860")
    print("   3. Go to 'Generate' tab → Enter text → Generate video")

if __name__ == "__main__":
    main()
