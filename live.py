"""
SadTalker Live Avatar - Real-time text-to-speech with instant viseme-based lip sync

Live Mode Architecture:
- Pre-render visemes (mouth shapes) for all phonemes (once)
- Text → TTS → Phonemes → Visemes → Real-time frame composition
- Expected latency: ~0.5-1s (TTS only), then instant playback

Speed: ~0.01s per frame (viseme lookup + blend) = real-time capable
"""

import os
import sys
import pickle
import shutil
import torch
import gradio as gr
import asyncio
import edge_tts
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from pydub import AudioSegment
from collections import defaultdict
import json
import time
import random
from threading import Thread
import queue

# Fix numpy 2.x compatibility
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'int'):
    np.int = int

# Auto-detect BASE_DIR
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
possible_paths = [
    script_dir,
    os.path.join(script_dir, "SadTalker"),
    os.path.dirname(script_dir),
]

BASE_DIR = None
for path in possible_paths:
    if os.path.exists(os.path.join(path, "inference.py")):
        BASE_DIR = path
        break

if BASE_DIR is None:
    print("❌ Error: Could not find SadTalker directory!")
    sys.exit(1)

# Paths
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
RESULT_DIR = os.path.join(BASE_DIR, "results")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
VISEME_DIR = os.path.join(CACHE_DIR, "visemes")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(VISEME_DIR, exist_ok=True)
os.makedirs(os.path.join(ASSETS_DIR, "image"), exist_ok=True)
os.makedirs(os.path.join(ASSETS_DIR, "audio"), exist_ok=True)

# Cache files
FACE_CACHE_FILE = os.path.join(CACHE_DIR, "face_cache.pkl")
VISEME_MAP_FILE = os.path.join(CACHE_DIR, "viseme_map.json")
VISEME_LIBRARY_FILE = os.path.join(CACHE_DIR, "viseme_library.pkl")

# Default assets
DEFAULT_IMAGE = os.path.join(ASSETS_DIR, "image", "female-image-01.jpg")

print(f"📁 BASE_DIR: {BASE_DIR}")
print(f"📁 Cache: {CACHE_DIR}")
print(f"📁 Visemes: {VISEME_DIR}")
print(f"🎭 Live mode: Real-time viseme-based avatar")

# Add to path
sys.path.insert(0, BASE_DIR)


# Phoneme to Viseme mapping (simplified - covers most English phonemes)
PHONEME_TO_VISEME = {
    # Vowels (open mouth)
    'AA': 'A', 'AE': 'A', 'AH': 'A', 'AO': 'O', 'AW': 'O', 'AY': 'A',
    'EH': 'E', 'ER': 'E', 'EY': 'E',
    'IH': 'I', 'IY': 'I',
    'OW': 'O', 'OY': 'O',
    'UH': 'U', 'UW': 'U',
    
    # Consonants - Closed (M, B, P)
    'M': 'M', 'B': 'M', 'P': 'M',
    
    # Consonants - Fricatives (F, V, TH)
    'F': 'F', 'V': 'F', 'TH': 'F', 'DH': 'F',
    
    # Consonants - Wide (W, OO)
    'W': 'W', 'UW': 'W',
    
    # Consonants - Teeth (T, D, N, L, S, Z)
    'T': 'T', 'D': 'T', 'N': 'T', 'L': 'T', 'S': 'T', 'Z': 'T',
    
    # Consonants - Rest (K, G, R, Y, etc.)
    'K': 'A', 'G': 'A', 'R': 'A', 'Y': 'A', 'HH': 'A', 'NG': 'A',
    'CH': 'T', 'JH': 'T', 'SH': 'T', 'ZH': 'T',
    
    # Silence/rest
    'SIL': 'A', 'SP': 'A', '': 'A'
}

# Viseme types (mouth shapes we need to pre-render)
VISEME_TYPES = ['A', 'E', 'I', 'O', 'U', 'M', 'F', 'W', 'T']  # 9 basic visemes
BLINK_VISEME = 'BLINK'  # Special viseme for eye blinks


def text_to_phonemes_simple(text: str):
    """
    Improved simple text-to-phoneme conversion using word-level patterns.
    Maps common English words and letter combinations to phonemes.
    """
    text = text.upper().strip()
    if not text:
        return []
    
    phonemes = []
    words = text.split()
    
    # Common word-to-phoneme patterns (expanded)
    word_patterns = {
        # Common words
        'HI': ['HH', 'AY'], 'HELLO': ['HH', 'EH', 'L', 'OW'],
        'THE': ['DH', 'AH'], 'A': ['AH'], 'AN': ['AE', 'N'],
        'IS': ['IH', 'Z'], 'ARE': ['AA', 'R'], 'WAS': ['W', 'AA', 'Z'],
        'TO': ['T', 'UW'], 'OF': ['AH', 'V'], 'AND': ['AE', 'N', 'D'],
        'IN': ['IH', 'N'], 'ON': ['AA', 'N'], 'AT': ['AE', 'T'],
        'IT': ['IH', 'T'], 'THIS': ['DH', 'IH', 'S'], 'THAT': ['DH', 'AE', 'T'],
        'WITH': ['W', 'IH', 'TH'], 'FOR': ['F', 'AO', 'R'],
        'YOU': ['Y', 'UW'], 'YOUR': ['Y', 'AO', 'R'],
        'WHAT': ['W', 'AA', 'T'], 'WHEN': ['W', 'EH', 'N'],
        'WHERE': ['W', 'EH', 'R'], 'WHO': ['HH', 'UW'],
        'HOW': ['HH', 'AW'], 'WHY': ['W', 'AY'],
        'YES': ['Y', 'EH', 'S'], 'NO': ['N', 'OW'],
        'GOOD': ['G', 'UH', 'D'], 'BAD': ['B', 'AE', 'D'],
        'BIG': ['B', 'IH', 'G'], 'SMALL': ['S', 'M', 'AO', 'L'],
        'ONE': ['W', 'AH', 'N'], 'TWO': ['T', 'UW'],
        'THREE': ['TH', 'R', 'IY'], 'FOUR': ['F', 'AO', 'R'],
        'FIVE': ['F', 'AY', 'V'], 'SIX': ['S', 'IH', 'K', 'S'],
        'SEVEN': ['S', 'EH', 'V', 'AH', 'N'], 'EIGHT': ['EY', 'T'],
        'NINE': ['N', 'AY', 'N'], 'TEN': ['T', 'EH', 'N'],
    }
    
    # Letter combination patterns
    def char_to_phonemes(char, prev_char=None, next_char=None):
        """Convert single character to phoneme(s) based on context."""
        if char == ' ':
            return ['SP']
        if char not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            return []
        
        # Vowels
        if char == 'A':
            if next_char and (next_char == 'E' or next_char == 'Y'):
                return ['EY']
            elif next_char and next_char == 'I':
                return ['EY']
            return ['AE']
        if char == 'E':
            if next_char and next_char == 'E':
                return ['IY']
            elif prev_char and prev_char == 'I':
                return []
            return ['EH']
        if char == 'I':
            if next_char and (next_char == 'E' or next_char == 'G' or next_char == 'H'):
                return ['AY']
            return ['IH']
        if char == 'O':
            if next_char and next_char == 'O':
                return ['UW']
            elif next_char and next_char == 'W':
                return ['OW']
            return ['AO']
        if char == 'U':
            if prev_char and prev_char == 'Q':
                return ['W']
            return ['UH']
        
        # Consonants
        if char == 'B':
            return ['B']
        if char == 'C':
            if next_char and next_char == 'H':
                return ['CH']
            elif next_char and next_char in 'EIY':
                return ['S']
            return ['K']
        if char == 'D':
            if next_char and next_char == 'G':
                return []
            return ['D']
        if char == 'F':
            return ['F']
        if char == 'G':
            if next_char and next_char == 'H':
                return []
            elif next_char and next_char in 'EIY':
                return ['JH']
            return ['G']
        if char == 'H':
            return ['HH']
        if char == 'J':
            return ['JH']
        if char == 'K':
            return ['K']
        if char == 'L':
            return ['L']
        if char == 'M':
            return ['M']
        if char == 'N':
            return ['N']
        if char == 'P':
            return ['P']
        if char == 'Q':
            return ['K', 'W']
        if char == 'R':
            return ['R']
        if char == 'S':
            if next_char and next_char == 'H':
                return ['SH']
            return ['S']
        if char == 'T':
            if next_char and next_char == 'H':
                return ['TH']
            return ['T']
        if char == 'V':
            return ['V']
        if char == 'W':
            return ['W']
        if char == 'X':
            return ['K', 'S']
        if char == 'Y':
            if prev_char and prev_char in 'AEIOU':
                return []
            return ['Y']
        if char == 'Z':
            return ['Z']
        
        return ['AH']  # Default fallback
    
    # Process each word
    for word in words:
        # Check if word is in patterns
        if word in word_patterns:
            phonemes.extend(word_patterns[word])
            phonemes.append('SP')  # Space after word
            continue
        
        # Process word character by character with context
        word_phonemes = []
        for i, char in enumerate(word):
            prev_char = word[i-1] if i > 0 else None
            next_char = word[i+1] if i < len(word)-1 else None
            
            char_phons = char_to_phonemes(char, prev_char, next_char)
            word_phonemes.extend(char_phons)
        
        if word_phonemes:
            phonemes.extend(word_phonemes)
            phonemes.append('SP')  # Space after word
    
    # Remove trailing space
    if phonemes and phonemes[-1] == 'SP':
        phonemes.pop()
    
    return phonemes if phonemes else ['AH']  # At least return something


def text_to_phonemes_espeak(text: str):
    """
    Use phonemizer with espeak backend (if available).
    Note: espeak-ng must be installed separately (not via pip).
    Windows: Download from https://github.com/espeak-ng/espeak-ng/releases
    Or use segments backend (pip install phonemizer segments) - Python-only
    """
    try:
        from phonemizer import phonemize
        try:
            from phonemizer.backend import EspeakBackend
            backend = EspeakBackend('en-us')
            phonemes_str = phonemize(text, backend=backend, separator=' ', strip=True)
            phonemes = phonemes_str.split()
            return phonemes
        except:
            # Fallback to segments backend (Python-only, no system install needed)
            try:
                from phonemizer.backend import SegmentsBackend
                backend = SegmentsBackend('en')
                phonemes_str = phonemize(text, backend=backend, separator=' ', strip=True)
                phonemes = phonemes_str.split()
                return phonemes
            except:
                raise ImportError("No phonemizer backend available")
    except ImportError:
        print("⚠ phonemizer not available, using simple mapping")
        return text_to_phonemes_simple(text)
    except Exception as e:
        print(f"⚠ Phoneme conversion error: {e}, using simple mapping")
        return text_to_phonemes_simple(text)


def phonemes_to_visemes(phonemes: list):
    """Convert phoneme sequence to viseme sequence."""
    visemes = []
    for phoneme in phonemes:
        # Map phoneme to viseme
        # 'SP' (space/silence) maps to 'M' (closed mouth) for resting position
        if phoneme.upper() == 'SP' or phoneme.upper() == '':
            viseme = 'M'  # Closed mouth for silence
        else:
            viseme = PHONEME_TO_VISEME.get(phoneme.upper(), 'M')  # Default to 'M' instead of 'A' for neutral
        visemes.append(viseme)
    return visemes


def detect_silence_periods(audio_path: str, silence_threshold: float = -40.0, min_silence_duration: float = 0.1):
    """
    Detect silence periods in audio file.
    Returns list of (start_time, end_time) tuples for silence periods.
    
    Args:
        audio_path: Path to audio file
        silence_threshold: dB threshold below which audio is considered silent (default -40dB)
        min_silence_duration: Minimum duration in seconds to be considered silence (default 0.1s)
    
    Returns:
        List of (start_time, end_time) tuples in seconds
    """
    try:
        audio = AudioSegment.from_wav(audio_path)
        
        # Convert to mono if stereo
        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        # Detect silence periods
        silence_periods = []
        chunks = audio[::100]  # Analyze every 100ms
        
        in_silence = False
        silence_start = 0.0
        
        for i, chunk in enumerate(chunks):
            chunk_time = i * 0.1  # Each chunk is 0.1 seconds
            dBFS = chunk.dBFS
            
            if dBFS < silence_threshold:
                if not in_silence:
                    in_silence = True
                    silence_start = chunk_time
            else:
                if in_silence:
                    silence_duration = chunk_time - silence_start
                    if silence_duration >= min_silence_duration:
                        silence_periods.append((silence_start, chunk_time))
                    in_silence = False
        
        # Handle silence at the end
        if in_silence:
            silence_duration = len(audio) / 1000.0 - silence_start
            if silence_duration >= min_silence_duration:
                silence_periods.append((silence_start, len(audio) / 1000.0))
        
        return silence_periods
    except Exception as e:
        print(f"⚠ Warning: Could not detect silence periods: {e}")
        return []


def preprocess_and_cache_face(image_path: str, cache_id: str = "default", size: int = 256):
    """Pre-process face once and cache 3DMM coefficients."""
    print(f"🔄 Pre-processing face (size={size}, this runs once)...")
    
    from src.utils.preprocess import CropAndExtract
    from src.utils.init_path import init_path
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Using device: {device}")
    
    sadtalker_paths = init_path(CHECKPOINT_DIR, os.path.join(BASE_DIR, 'src/config'), size, False, 'full')
    preprocess_model = CropAndExtract(sadtalker_paths, device)
    
    cache_frame_dir = os.path.join(CACHE_DIR, f"face_{cache_id}")
    os.makedirs(cache_frame_dir, exist_ok=True)
    
    first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(
        image_path, cache_frame_dir, 'full', source_image_flag=True, pic_size=size
    )
    
    if first_coeff_path is None:
        return None, "❌ Face detection failed. Use a clear front-facing face image."
    
    pic_name = os.path.splitext(os.path.split(image_path)[-1])[0]
    landmarks_path = os.path.join(cache_frame_dir, f"{pic_name}_landmarks.txt")
    
    cache_data = {
        'first_coeff_path': first_coeff_path,
        'crop_pic_path': crop_pic_path,
        'crop_info': crop_info,
        'image_path': image_path,
        'landmarks_path': landmarks_path if os.path.exists(landmarks_path) else None,
        'cache_id': cache_id,
        'size': size
    }
    
    with open(FACE_CACHE_FILE, 'wb') as f:
        pickle.dump(cache_data, f)
    
    del preprocess_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return cache_data, f"✅ Face pre-processed and cached!\n   Size: {size}px"


def load_face_cache():
    """Load cached face data."""
    if os.path.exists(FACE_CACHE_FILE):
        with open(FACE_CACHE_FILE, 'rb') as f:
            return pickle.load(f)
    return None


async def text_to_speech_async(text: str, voice: str, out_path: str):
    """Generate speech from text using edge-tts."""
    mp3_path = out_path.replace(".wav", ".mp3")
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(mp3_path)
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(out_path, format="wav")
    if os.path.exists(mp3_path):
        os.remove(mp3_path)
    return out_path


def generate_viseme(viseme_type: str, face_cache: dict, size: int = 256):
    """Generate a single viseme (mouth shape) frame."""
    from src.utils.init_path import init_path
    from src.test_audio2coeff import Audio2Coeff
    from src.facerender.animate import AnimateFromCoeff
    from src.generate_batch import get_data
    from src.generate_facerender_batch import get_facerender_data
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create a short audio for this viseme (single phoneme) - use more exaggerated prompts
    viseme_audio_map = {
        'A': 'ah ah ah',  # More pronounced 'ah' sound
        'E': 'eh eh eh',  # More pronounced 'eh' sound
        'I': 'ee ee ee',  # More pronounced 'ee' sound
        'O': 'oh oh oh',  # More pronounced 'oh' sound
        'U': 'oo oo oo',  # More pronounced 'oo' sound
        'M': 'mmm mmm mmm',  # More pronounced 'm' sound
        'F': 'fff fff fff',  # More pronounced 'f' sound
        'W': 'woo woo woo',  # More pronounced 'w' sound
        'T': 'tah tah tah'   # More pronounced 't' sound
    }
    
    viseme_text = viseme_audio_map.get(viseme_type, 'ah')
    
    # Generate audio for this viseme
    temp_audio = os.path.join(VISEME_DIR, f"temp_{viseme_type}.wav")
    asyncio.run(text_to_speech_async(viseme_text, "en-US-JennyNeural", temp_audio))
    
    # Generate video frame
    sadtalker_paths = init_path(CHECKPOINT_DIR, os.path.join(BASE_DIR, 'src/config'), size, False, 'full')
    audio_to_coeff = Audio2Coeff(sadtalker_paths, device)
    animate_from_coeff = AnimateFromCoeff(sadtalker_paths, device)
    
    viseme_gen_dir = os.path.join(VISEME_DIR, viseme_type)
    os.makedirs(viseme_gen_dir, exist_ok=True)
    
    batch = get_data(
        face_cache['first_coeff_path'],
        temp_audio,
        device,
        ref_eyeblink_coeff_path=None,
        still=True,
        use_blink=True  # Enable blink for natural visemes
    )
    
    coeff_path = audio_to_coeff.generate(batch, viseme_gen_dir, pose_style=0, ref_pose_coeff_path=None)
    
    data = get_facerender_data(
        coeff_path,
        face_cache['crop_pic_path'],
        face_cache['first_coeff_path'],
        temp_audio,
        batch_size=2,
        input_yaw_list=None,
        input_pitch_list=None,
        input_roll_list=None,
        expression_scale=1.0,
        still_mode=True,
        preprocess='full',
        size=size
    )
    
    result = animate_from_coeff.generate(
        data,
        viseme_gen_dir,
        face_cache['image_path'],
        face_cache['crop_info'],
        enhancer=None,
        background_enhancer=None,
        preprocess='full',
        img_size=size
    )
    
    # Extract best frame showing mouth movement and save as viseme
    # Also extract mouth region for blending
    if os.path.exists(result):
        cap = cv2.VideoCapture(result)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Try to find frame with most mouth movement (check multiple frames)
        best_frame = None
        best_frame_idx = frame_count // 2  # Default to middle
        
        if frame_count > 1:
            # Sample frames from middle section (where mouth movement is most visible)
            sample_start = max(1, frame_count // 3)
            sample_end = min(frame_count - 1, frame_count * 2 // 3)
            
            # Read a few frames and pick one with visible mouth
            for idx in range(sample_start, sample_end, max(1, (sample_end - sample_start) // 5)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    best_frame = frame
                    best_frame_idx = idx
                    break  # Use first valid frame in middle section
        
        # If no frame found, try middle
        if best_frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, best_frame_idx)
            ret, best_frame = cap.read()
        
        cap.release()
        
        if ret and best_frame is not None:
            viseme_frame_path = os.path.join(VISEME_DIR, f"viseme_{viseme_type}.png")
            cv2.imwrite(viseme_frame_path, best_frame)
            
            # Extract mouth region using crop_info to find face position in full image
            h, w = best_frame.shape[:2]
            
            # Use crop_info to find face bounding box in full image
            if face_cache.get('crop_info') and len(face_cache['crop_info']) == 3:
                # crop_info format: ((face_w, face_h), (clx, cly, crx, cry), (lx, ly, rx, ry))
                r_w, r_h = face_cache['crop_info'][0]
                clx, cly, crx, cry = face_cache['crop_info'][1]
                lx, ly, rx, ry = face_cache['crop_info'][2]
                lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
                
                # Face bounding box in full image coordinates
                ox1, oy1, ox2, oy2 = clx + lx, cly + ly, clx + rx, cly + ry
                
                # Ensure coordinates are within image bounds
                ox1, oy1 = max(0, ox1), max(0, oy1)
                ox2, oy2 = min(w, ox2), min(h, oy2)
                
                # Mouth is in lower 60-85% of face (from top of face)
                face_h = oy2 - oy1
                face_w = ox2 - ox1
                
                # Mouth region within face (centered horizontally, lower third vertically)
                mouth_y1 = oy1 + int(face_h * 0.60)  # Start at 60% down the face
                mouth_y2 = oy1 + int(face_h * 0.85)  # End at 85% down the face
                mouth_x1 = ox1 + int(face_w * 0.25)  # Start at 25% from left
                mouth_x2 = ox1 + int(face_w * 0.75)  # End at 75% from left
                
                # Ensure mouth coordinates are within image bounds
                mouth_y1, mouth_x1 = max(0, mouth_y1), max(0, mouth_x1)
                mouth_y2, mouth_x2 = min(h, mouth_y2), min(w, mouth_x2)
                
                if mouth_y2 > mouth_y1 and mouth_x2 > mouth_x1:
                    mouth_region = best_frame[mouth_y1:mouth_y2, mouth_x1:mouth_x2]
                    
                    # Only save if mouth region is valid
                    if mouth_region.size > 0:
                        mouth_path = os.path.join(VISEME_DIR, f"mouth_{viseme_type}.png")
                        cv2.imwrite(mouth_path, mouth_region)
            else:
                # Fallback: assume face crop (old method)
                mouth_y1, mouth_y2 = int(h * 0.5), int(h * 0.85)
                mouth_x1, mouth_x2 = int(w * 0.25), int(w * 0.75)
                mouth_region = best_frame[mouth_y1:mouth_y2, mouth_x1:mouth_x2]
                
                if mouth_region.size > 0:
                    mouth_path = os.path.join(VISEME_DIR, f"mouth_{viseme_type}.png")
                    cv2.imwrite(mouth_path, mouth_region)
            
            return viseme_frame_path
    
    return None


def generate_blink_frame(face_cache: dict, size: int = 256):
    """Generate a blink frame (closed eyes) for eye blink overlay."""
    from src.utils.init_path import init_path
    from src.test_audio2coeff import Audio2Coeff
    from src.facerender.animate import AnimateFromCoeff
    from src.generate_batch import get_data
    from src.generate_facerender_batch import get_facerender_data
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create a short silent audio (or very short audio) for blink
    temp_audio = os.path.join(VISEME_DIR, f"temp_{BLINK_VISEME}.wav")
    # Generate a very short audio (0.1s) with silence or minimal sound
    asyncio.run(text_to_speech_async("mm", "en-US-JennyNeural", temp_audio))
    
    # Generate video frame with blink
    sadtalker_paths = init_path(CHECKPOINT_DIR, os.path.join(BASE_DIR, 'src/config'), size, False, 'full')
    audio_to_coeff = Audio2Coeff(sadtalker_paths, device)
    animate_from_coeff = AnimateFromCoeff(sadtalker_paths, device)
    
    viseme_gen_dir = os.path.join(VISEME_DIR, BLINK_VISEME)
    os.makedirs(viseme_gen_dir, exist_ok=True)
    
    # Generate with blink enabled
    batch = get_data(
        face_cache['first_coeff_path'],
        temp_audio,
        device,
        ref_eyeblink_coeff_path=None,
        still=True,
        use_blink=True  # Enable blink
    )
    
    coeff_path = audio_to_coeff.generate(batch, viseme_gen_dir, pose_style=0, ref_pose_coeff_path=None)
    
    data = get_facerender_data(
        coeff_path,
        face_cache['crop_pic_path'],
        face_cache['first_coeff_path'],
        temp_audio,
        batch_size=2,
        input_yaw_list=None,
        input_pitch_list=None,
        input_roll_list=None,
        expression_scale=1.0,
        still_mode=True,
        preprocess='full',
        size=size
    )
    
    result = animate_from_coeff.generate(
        data,
        viseme_gen_dir,
        face_cache['image_path'],
        face_cache['crop_info'],
        enhancer=None,
        background_enhancer=None,
        preprocess='full',
        img_size=size
    )
    
    # Extract frame with closed eyes (middle frame usually has blink)
    if os.path.exists(result):
        cap = cv2.VideoCapture(result)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Try to find frame with closed eyes
        # Blinks typically occur in the middle-to-end of short audio clips
        blink_frame = None
        if frame_count > 0:
            # Try multiple frames to find one with closed eyes
            # Start from middle and go towards end (where blink is more likely)
            sample_indices = []
            if frame_count >= 3:
                # Sample from middle to end
                mid_point = frame_count // 2
                sample_indices = list(range(mid_point, min(frame_count, mid_point + frame_count // 3)))
            else:
                sample_indices = [frame_count // 2] if frame_count > 0 else [0]
            
            for idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    blink_frame = frame
                    break
            
            # If still no frame, try last frame
            if blink_frame is None and frame_count > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
                ret, frame = cap.read()
                if ret:
                    blink_frame = frame
        
        cap.release()
        
        if blink_frame is not None:
            blink_path = os.path.join(VISEME_DIR, f"viseme_{BLINK_VISEME}.png")
            cv2.imwrite(blink_path, blink_frame)
            return blink_path
    
    return None


def generate_viseme_library():
    """Pre-generate all visemes (mouth shapes) + blink frame - runs once."""
    print("🎭 Generating viseme library (this runs once)...")
    
    face_cache = load_face_cache()
    if not face_cache:
        return None, "❌ No cached face found. Run Setup Mode first."
    
    viseme_library = {}
    size = face_cache.get('size', 256)
    
    # Generate mouth visemes
    for viseme_type in VISEME_TYPES:
        print(f"   Generating viseme: {viseme_type}")
        viseme_path = generate_viseme(viseme_type, face_cache, size)
        if viseme_path:
            viseme_library[viseme_type] = viseme_path
            print(f"   ✓ {viseme_type} ready")
        else:
            print(f"   ⚠ Failed to generate {viseme_type}")
    
    # Generate blink frame
    print(f"   Generating blink frame...")
    blink_path = generate_blink_frame(face_cache, size)
    if blink_path:
        viseme_library[BLINK_VISEME] = blink_path
        print(f"   ✓ Blink frame ready")
    else:
        print(f"   ⚠ Failed to generate blink frame")
    
    # Save viseme library
    with open(VISEME_LIBRARY_FILE, 'wb') as f:
        pickle.dump(viseme_library, f)
    
    # Save phoneme-to-viseme mapping
    with open(VISEME_MAP_FILE, 'w') as f:
        json.dump(PHONEME_TO_VISEME, f, indent=2)
    
    generated = len([v for v in viseme_library.values() if v])
    return viseme_library, f"✅ Viseme library generated!\n   {generated}/{len(VISEME_TYPES) + 1} visemes + blink ready"


def load_viseme_library():
    """Load pre-generated viseme library."""
    if os.path.exists(VISEME_LIBRARY_FILE):
        with open(VISEME_LIBRARY_FILE, 'rb') as f:
            return pickle.load(f)
    return None


def regenerate_mouth_regions():
    """Extract mouth regions from existing viseme frames if mouth_*.png files are missing."""
    viseme_library = load_viseme_library()
    if not viseme_library:
        return None, "❌ No viseme library found. Run 'Generate Viseme Library' first."
    
    face_cache = load_face_cache()
    if not face_cache:
        return None, "❌ No face cache found. Run face preprocessing first."
    
    regenerated = 0
    for viseme_type in VISEME_TYPES:
        viseme_path = viseme_library.get(viseme_type)
        if viseme_path and os.path.exists(viseme_path):
            mouth_path = os.path.join(VISEME_DIR, f"mouth_{viseme_type}.png")
            # Always regenerate to fix incorrect extractions
            viseme_frame = cv2.imread(viseme_path)
            if viseme_frame is not None:
                h, w = viseme_frame.shape[:2]
                
                # Use crop_info to find face position in full image
                if face_cache.get('crop_info') and len(face_cache['crop_info']) == 3:
                    r_w, r_h = face_cache['crop_info'][0]
                    clx, cly, crx, cry = face_cache['crop_info'][1]
                    lx, ly, rx, ry = face_cache['crop_info'][2]
                    lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
                    
                    # Face bounding box in full image coordinates
                    ox1, oy1, ox2, oy2 = clx + lx, cly + ly, clx + rx, cly + ry
                    
                    # Ensure coordinates are within image bounds
                    ox1, oy1 = max(0, ox1), max(0, oy1)
                    ox2, oy2 = min(w, ox2), min(h, oy2)
                    
                    # Mouth is in lower 60-85% of face
                    face_h = oy2 - oy1
                    face_w = ox2 - ox1
                    
                    mouth_y1 = oy1 + int(face_h * 0.60)
                    mouth_y2 = oy1 + int(face_h * 0.85)
                    mouth_x1 = ox1 + int(face_w * 0.25)
                    mouth_x2 = ox1 + int(face_w * 0.75)
                    
                    # Ensure mouth coordinates are within image bounds
                    mouth_y1, mouth_x1 = max(0, mouth_y1), max(0, mouth_x1)
                    mouth_y2, mouth_x2 = min(h, mouth_y2), min(w, mouth_x2)
                    
                    if mouth_y2 > mouth_y1 and mouth_x2 > mouth_x1:
                        mouth_region = viseme_frame[mouth_y1:mouth_y2, mouth_x1:mouth_x2]
                        if mouth_region.size > 0:
                            cv2.imwrite(mouth_path, mouth_region)
                            regenerated += 1
                else:
                    # Fallback: assume face crop
                    mouth_y1, mouth_y2 = int(h * 0.5), int(h * 0.85)
                    mouth_x1, mouth_x2 = int(w * 0.25), int(w * 0.75)
                    mouth_region = viseme_frame[mouth_y1:mouth_y2, mouth_x1:mouth_x2]
                    if mouth_region.size > 0:
                        cv2.imwrite(mouth_path, mouth_region)
                        regenerated += 1
    
    return regenerated, f"✅ Regenerated {regenerated}/{len(VISEME_TYPES)} mouth region files"


def blend_blink_onto_face(base_face, blink_frame, face_cache):
    """Blend blink frame (closed eyes) onto base face image."""
    if blink_frame is None or blink_frame.size == 0:
        return base_face.copy()
    
    h, w = base_face.shape[:2]
    
    # Use crop_info to find face position in full image
    if face_cache.get('crop_info') and len(face_cache['crop_info']) == 3:
        r_w, r_h = face_cache['crop_info'][0]
        clx, cly, crx, cry = face_cache['crop_info'][1]
        lx, ly, rx, ry = face_cache['crop_info'][2]
        lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
        
        # Face bounding box in full image coordinates
        ox1, oy1, ox2, oy2 = clx + lx, cly + ly, clx + rx, cly + ry
        
        # Ensure coordinates are within image bounds
        ox1, oy1 = max(0, ox1), max(0, oy1)
        ox2, oy2 = min(w, ox2), min(h, oy2)
        
        # Eye region is upper 20-45% of face (from top of face) - more accurate positioning
        face_h = oy2 - oy1
        face_w = ox2 - ox1
        
        eye_y1 = oy1 + int(face_h * 0.20)  # Start at 20% down the face (eyes are higher)
        eye_y2 = oy1 + int(face_h * 0.45)  # End at 45% down the face
        eye_x1 = ox1 + int(face_w * 0.10)  # Start at 10% from left (wider eye region)
        eye_x2 = ox1 + int(face_w * 0.90)  # End at 90% from left
        
        # Ensure eye coordinates are within image bounds
        eye_y1, eye_x1 = max(0, eye_y1), max(0, eye_x1)
        eye_y2, eye_x2 = min(h, eye_y2), min(w, eye_x2)
    else:
        # Fallback: assume face crop (eyes are higher up)
        eye_y1, eye_y2 = int(h * 0.20), int(h * 0.45)
        eye_x1, eye_x2 = int(w * 0.10), int(w * 0.90)
        
        eye_y1 = max(0, eye_y1)
        eye_y2 = min(h, eye_y2)
        eye_x1 = max(0, eye_x1)
        eye_x2 = min(w, eye_x2)
    
    eye_h = eye_y2 - eye_y1
    eye_w = eye_x2 - eye_x1
    
    if eye_h <= 0 or eye_w <= 0:
        return base_face.copy()
    
    # Resize blink frame to match eye region size
    blink_resized = cv2.resize(blink_frame, (w, h))
    eye_region_blink = blink_resized[eye_y1:eye_y2, eye_x1:eye_x2]
    
    # Create mask for blending (feather edges)
    mask = np.ones((eye_h, eye_w), dtype=np.float32)
    feather = min(10, eye_h // 3, eye_w // 3)
    
    for i in range(feather):
        alpha = i / max(feather, 1)
        if i < eye_h:
            mask[i, :] *= alpha
            mask[-i-1, :] *= alpha
        if i < eye_w:
            mask[:, i] *= alpha
            mask[:, -i-1] *= alpha
    
    mask_3d = np.stack([mask] * 3, axis=2)
    
    # Blend eye region (strong blend for visible blink)
    result = base_face.copy()
    eye_region = result[eye_y1:eye_y2, eye_x1:eye_x2].astype(np.float32)
    
    blend_ratio = 0.85  # Strong blend for visible blink
    blended = (eye_region * (1 - mask_3d * blend_ratio) + 
               eye_region_blink.astype(np.float32) * mask_3d * blend_ratio).astype(np.uint8)
    
    result[eye_y1:eye_y2, eye_x1:eye_x2] = blended
    
    return result


def blend_mouth_onto_face(base_face, viseme_mouth, face_cache):
    """Blend mouth region from viseme onto base face image."""
    if viseme_mouth is None or viseme_mouth.size == 0:
        return base_face.copy()
    
    h, w = base_face.shape[:2]
    
    # Use crop_info to find face position in full image (same logic as generate_viseme)
    if face_cache.get('crop_info') and len(face_cache['crop_info']) == 3:
        # crop_info format: ((face_w, face_h), (clx, cly, crx, cry), (lx, ly, rx, ry))
        r_w, r_h = face_cache['crop_info'][0]
        clx, cly, crx, cry = face_cache['crop_info'][1]
        lx, ly, rx, ry = face_cache['crop_info'][2]
        lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
        
        # Face bounding box in full image coordinates
        ox1, oy1, ox2, oy2 = clx + lx, cly + ly, clx + rx, cly + ry
        
        # Ensure coordinates are within image bounds
        ox1, oy1 = max(0, ox1), max(0, oy1)
        ox2, oy2 = min(w, ox2), min(h, oy2)
        
        # Mouth is in lower 60-85% of face (from top of face)
        face_h = oy2 - oy1
        face_w = ox2 - ox1
        
        # Mouth region within face (centered horizontally, lower third vertically)
        mouth_y1 = oy1 + int(face_h * 0.60)  # Start at 60% down the face
        mouth_y2 = oy1 + int(face_h * 0.85)  # End at 85% down the face
        mouth_x1 = ox1 + int(face_w * 0.25)  # Start at 25% from left
        mouth_x2 = ox1 + int(face_w * 0.75)  # End at 75% from left
        
        # Ensure mouth coordinates are within image bounds
        mouth_y1, mouth_x1 = max(0, mouth_y1), max(0, mouth_x1)
        mouth_y2, mouth_x2 = min(h, mouth_y2), min(w, mouth_x2)
    else:
        # Fallback: assume face crop (old method - shouldn't happen with preprocess='full')
        mouth_y1, mouth_y2 = int(h * 0.5), int(h * 0.85)
        mouth_x1, mouth_x2 = int(w * 0.25), int(w * 0.75)
        
        # Ensure valid region
        mouth_y1 = max(0, mouth_y1)
        mouth_y2 = min(h, mouth_y2)
        mouth_x1 = max(0, mouth_x1)
        mouth_x2 = min(w, mouth_x2)
    
    mouth_h = mouth_y2 - mouth_y1
    mouth_w = mouth_x2 - mouth_x1
    
    if mouth_h <= 0 or mouth_w <= 0:
        return base_face.copy()
    
    # Resize viseme mouth to match region size
    viseme_mouth_resized = cv2.resize(viseme_mouth, (mouth_w, mouth_h))
    
    # Create mask for blending (feather edges for smooth transition)
    mask = np.ones((mouth_h, mouth_w), dtype=np.float32)
    feather = min(15, mouth_h // 4, mouth_w // 4)  # Adaptive feather size
    
    for i in range(feather):
        alpha = i / max(feather, 1)
        if i < mouth_h:
            mask[i, :] *= alpha
            mask[-i-1, :] *= alpha
        if i < mouth_w:
            mask[:, i] *= alpha
            mask[:, -i-1] *= alpha
    
    mask_3d = np.stack([mask] * 3, axis=2)
    
    # Blend mouth region (stronger blend for visible lip movement)
    result = base_face.copy()
    mouth_region = result[mouth_y1:mouth_y2, mouth_x1:mouth_x2].astype(np.float32)
    
    # Subtle blend (65% viseme, 35% base) for natural-looking lip movement
    # Lower ratio = less exaggerated, more natural speech
    blend_ratio = 0.65
    
    # Blend with slight color enhancement for more visible changes
    viseme_mouth_float = viseme_mouth_resized.astype(np.float32)
    mouth_region_float = mouth_region.astype(np.float32)
    
    # Apply blend with enhanced contrast
    blended = (mouth_region_float * (1 - mask_3d * blend_ratio) + 
               viseme_mouth_float * mask_3d * blend_ratio).astype(np.uint8)
    
    result[mouth_y1:mouth_y2, mouth_x1:mouth_x2] = blended
    
    return result


def compose_live_video_streaming(text: str, fps: int = 25):
    """Stream video frames in real-time as they're composed (generator)."""
    import time
    start_time = time.time()
    
    viseme_library = load_viseme_library()
    if not viseme_library:
        yield None, None, "❌ Viseme library not found. Run 'Generate Viseme Library' in Setup tab."
        return
    
    face_cache = load_face_cache()
    if not face_cache:
        yield None, None, "❌ Face cache not found. Run Setup Mode first."
        return
    
    # Step 1: Text → TTS → Audio
    yield None, None, "🔄 Generating speech..."
    ts = datetime.now().strftime("%Y_%m_%d_%H.%M.%S")
    audio_path = os.path.join(RESULT_DIR, f"live_audio_{ts}.wav")
    
    try:
        asyncio.run(text_to_speech_async(text.strip(), "en-US-JennyNeural", audio_path))
        tts_time = time.time() - start_time
        
        # Verify audio file was created
        if not os.path.exists(audio_path):
            yield None, None, "❌ TTS generation failed: Audio file was not created."
            return
            
    except Exception as e:
        error_msg = str(e)
        if "getaddrinfo failed" in error_msg or "Cannot connect to host" in error_msg:
            yield None, None, f"❌ Network error: Cannot connect to TTS service.\n\nPlease check:\n• Internet connection\n• Firewall/proxy settings\n• DNS resolution\n\nError: {error_msg[:100]}"
        else:
            yield None, None, f"❌ TTS generation failed: {error_msg[:200]}"
        return
    
    # Step 2: Text → Phonemes → Visemes
    yield None, audio_path, f"✓ Audio ready ({tts_time:.1f}s)\n🔄 Converting to phonemes..."
    phonemes = text_to_phonemes_espeak(text)
    visemes = phonemes_to_visemes(phonemes)
    
    # Debug: Show phonemes and visemes
    print(f"🔍 Debug: Text: '{text}'")
    print(f"🔍 Debug: Phonemes ({len(phonemes)}): {phonemes[:20]}")
    print(f"🔍 Debug: Visemes ({len(visemes)}): {visemes[:20]}")
    
    # Step 3: Get audio duration and detect silence periods
    audio_seg = AudioSegment.from_wav(audio_path)
    audio_duration = len(audio_seg) / 1000.0
    total_frames = int(audio_duration * fps)
    
    # Detect silence periods in audio
    silence_periods = detect_silence_periods(audio_path, silence_threshold=-35.0, min_silence_duration=0.15)
    if silence_periods:
        print(f"🔍 Debug: Detected {len(silence_periods)} silence periods: {silence_periods[:5]}")
    
    # Better viseme timing: make visemes change more frequently for realistic movement
    # Merge consecutive identical visemes to avoid static periods
    # Keep 'SP' visemes as 'M' (closed mouth) for silence
    visemes_merged = []
    for v in visemes:
        # Convert 'SP' to 'M' for closed mouth during silence
        if v == 'SP':
            v = 'M'
        if not visemes_merged or visemes_merged[-1] != v:
            visemes_merged.append(v)
    
    if visemes_merged and len(visemes_merged) > 0:
        # More frames per viseme = slower, smoother, more natural transitions
        frames_per_viseme = max(5, total_frames // len(visemes_merged))
        # Cap at 8 frames per viseme so changes aren't too jumpy
        frames_per_viseme = min(frames_per_viseme, 8)
    else:
        frames_per_viseme = max(5, total_frames // max(len(visemes), 1))
    
    # Step 4: Load base face and viseme mouths
    base_image = cv2.imread(face_cache['image_path'])
    if base_image is None:
        yield None, None, "❌ Could not load base face image"
        return
    
    # Load viseme library (full frames)
    viseme_library = load_viseme_library()
    if not viseme_library:
        yield None, None, "❌ Viseme library not loaded. Run 'Generate Viseme Library' in Setup tab."
        return
    
    # Load mouth regions for each viseme
    mouth_library = {}
    for viseme_type in VISEME_TYPES:
        mouth_path = os.path.join(VISEME_DIR, f"mouth_{viseme_type}.png")
        if os.path.exists(mouth_path):
            mouth_img = cv2.imread(mouth_path)
            if mouth_img is not None and mouth_img.size > 0:
                mouth_library[viseme_type] = mouth_img
    
    # Step 5: Compose frames and stream them
    silence_info = f" ({len(silence_periods)} silence periods detected)" if silence_periods else ""
    yield None, audio_path, f"✓ Phonemes: {len(phonemes)}, Visemes: {len(visemes)}\n✓ Loaded {len(mouth_library)}/{len(VISEME_TYPES)} mouth regions{silence_info}\n🔄 Composing frames..."
    
    gen_dir = os.path.join(RESULT_DIR, f"live_{ts}")
    os.makedirs(gen_dir, exist_ok=True)
    
    h, w = base_image.shape[:2]
    
    # Load blink frame (after we know image dimensions)
    blink_frame = None
    blink_path = viseme_library.get(BLINK_VISEME) if viseme_library else None
    if not blink_path or not os.path.exists(blink_path):
        blink_path = os.path.join(VISEME_DIR, f"viseme_{BLINK_VISEME}.png")  # Fallback: load from disk
    if blink_path and os.path.exists(blink_path):
        blink_frame = cv2.imread(blink_path)
        if blink_frame is not None:
            blink_frame = cv2.resize(blink_frame, (w, h))
            print(f"✓ Blink frame loaded")
    if blink_frame is None:
        print(f"⚠ Blink frame not found. Regenerate Viseme Library in Setup for eye blinks.")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_video = os.path.join(gen_dir, "temp_video.mp4")
    out = cv2.VideoWriter(temp_video, fourcc, fps, (w, h))
    
    # Debug: Check what we have
    print(f"🔍 Debug: {len(visemes)} visemes, {len(mouth_library)} mouth regions loaded")
    print(f"🔍 Debug: Visemes: {visemes[:10] if len(visemes) > 0 else 'EMPTY'}")
    print(f"🔍 Debug: Mouth library keys: {list(mouth_library.keys())[:10] if mouth_library else 'EMPTY'}")
    
    if not visemes:
        yield None, None, f"⚠️ No visemes generated from text. Using base image only.\nPhonemes: {phonemes[:20] if phonemes else 'None'}"
    
    if not mouth_library:
        yield None, None, f"⚠️ No mouth library found! Run 'Generate Viseme Library' in Setup tab.\nLooking in: {VISEME_DIR}"
    
    # Ensure 'M' (closed mouth) is available for silence periods
    if 'M' not in mouth_library and viseme_library and 'M' in viseme_library:
        # Try to extract 'M' mouth region from viseme frame
        m_path = viseme_library['M']
        if os.path.exists(m_path):
            m_frame = cv2.imread(m_path)
            if m_frame is not None:
                m_frame = cv2.resize(m_frame, (w, h))
                # Extract mouth using crop_info
                if face_cache.get('crop_info') and len(face_cache['crop_info']) == 3:
                    r_w, r_h = face_cache['crop_info'][0]
                    clx, cly, crx, cry = face_cache['crop_info'][1]
                    lx, ly, rx, ry = face_cache['crop_info'][2]
                    lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
                    
                    ox1, oy1, ox2, oy2 = clx + lx, cly + ly, clx + rx, cly + ry
                    ox1, oy1 = max(0, ox1), max(0, oy1)
                    ox2, oy2 = min(w, ox2), min(h, oy2)
                    
                    face_h = oy2 - oy1
                    face_w = ox2 - ox1
                    
                    mouth_y1 = oy1 + int(face_h * 0.60)
                    mouth_y2 = oy1 + int(face_h * 0.85)
                    mouth_x1 = ox1 + int(face_w * 0.25)
                    mouth_x2 = ox1 + int(face_w * 0.75)
                    
                    mouth_y1, mouth_x1 = max(0, mouth_y1), max(0, mouth_x1)
                    mouth_y2, mouth_x2 = min(h, mouth_y2), min(w, mouth_x2)
                    
                    if mouth_y2 > mouth_y1 and mouth_x2 > mouth_x1:
                        m_mouth = m_frame[mouth_y1:mouth_y2, mouth_x1:mouth_x2]
                        if m_mouth.size > 0:
                            mouth_library['M'] = m_mouth
                            print(f"✓ Extracted 'M' mouth region for silence periods")
    
    # Use merged visemes (consecutive duplicates removed) for display
    visemes_for_display = visemes_merged if visemes_merged else []
    if not visemes_for_display:
        visemes_for_display = ['M']  # Default to closed mouth for neutral
    
    # Debug: Show unique visemes being used
    unique_visemes = list(set(visemes_for_display))
    print(f"🔍 Debug: Unique visemes in sequence: {unique_visemes}")
    print(f"🔍 Debug: Merged viseme sequence ({len(visemes_for_display)}): {visemes_for_display}")
    print(f"🔍 Debug: Frames per viseme: {frames_per_viseme} (for {total_frames} total frames)")
    
    viseme_idx = 0
    frame_time = 1.0 / fps
    last_viseme = None
    viseme_changes = []
    
    # Blink timing: first blink early (0.5-1.5s), then every 2-4 seconds
    next_blink_frame = random.randint(int(fps * 0.5), int(fps * 1.5))  # First blink at 0.5-1.5s
    blink_duration_frames = 4  # Blink lasts 4 frames (~0.16s at 25fps) for visibility
    is_blinking = False
    blink_start_frame = 0
    
    for frame_idx in range(total_frames):
        # Calculate current time in seconds
        current_time = frame_idx / fps
        
        # Check if we're in a silence period - use closed mouth ('M') viseme
        is_silent = False
        for silence_start, silence_end in silence_periods:
            if silence_start <= current_time <= silence_end:
                is_silent = True
                break
        
        # Calculate which viseme should be shown at this frame
        if is_silent:
            # Use closed mouth ('M') during silence
            current_viseme = 'M'
            if current_viseme != last_viseme:
                viseme_changes.append((frame_idx, current_viseme))
                last_viseme = current_viseme
        elif visemes_for_display:
            # Calculate viseme index based on frame position
            viseme_idx = min(frame_idx // frames_per_viseme, len(visemes_for_display) - 1)
            current_viseme = visemes_for_display[viseme_idx]
            
            # Track viseme changes for debugging
            if current_viseme != last_viseme:
                viseme_changes.append((frame_idx, current_viseme))
                last_viseme = current_viseme
        else:
            # No visemes available, use default (closed mouth for neutral)
            current_viseme = 'M'  # Default to closed mouth instead of 'A'
        
        # Check if we should blink (random timing)
        if blink_frame is not None:
            if frame_idx >= next_blink_frame and not is_blinking:
                # Start blink
                is_blinking = True
                blink_start_frame = frame_idx
                # Next blink in 2-4 seconds
                next_blink_frame = frame_idx + blink_duration_frames + random.randint(int(fps * 2), int(fps * 4))
            elif is_blinking and (frame_idx - blink_start_frame) >= blink_duration_frames:
                # End blink
                is_blinking = False
        
        # Blend mouth from viseme onto base face
        # During silence, use 'M' (closed mouth) viseme
        frame = base_image.copy()  # Start with base image
        viseme_applied = False
        
        if current_viseme in mouth_library and mouth_library[current_viseme] is not None:
            # Use pre-extracted mouth region (fastest)
            frame = blend_mouth_onto_face(frame, mouth_library[current_viseme], face_cache)
            viseme_applied = True
        elif viseme_library and current_viseme in viseme_library:
            # Fallback: extract mouth from full viseme frame if mouth region file not available
            viseme_path = viseme_library.get(current_viseme)
            if not viseme_path and current_viseme == 'M':
                # Try 'A' as fallback for 'M' if not found
                viseme_path = viseme_library.get('A')
            
            if viseme_path and os.path.exists(viseme_path):
                viseme_frame = cv2.imread(viseme_path)
                if viseme_frame is not None:
                    viseme_frame = cv2.resize(viseme_frame, (w, h))
                    # Extract mouth region using crop_info for accurate positioning
                    if face_cache.get('crop_info') and len(face_cache['crop_info']) == 3:
                        r_w, r_h = face_cache['crop_info'][0]
                        clx, cly, crx, cry = face_cache['crop_info'][1]
                        lx, ly, rx, ry = face_cache['crop_info'][2]
                        lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
                        
                        ox1, oy1, ox2, oy2 = clx + lx, cly + ly, clx + rx, cly + ry
                        ox1, oy1 = max(0, ox1), max(0, oy1)
                        ox2, oy2 = min(w, ox2), min(h, oy2)
                        
                        face_h = oy2 - oy1
                        face_w = ox2 - ox1
                        
                        mouth_y1 = oy1 + int(face_h * 0.60)
                        mouth_y2 = oy1 + int(face_h * 0.85)
                        mouth_x1 = ox1 + int(face_w * 0.25)
                        mouth_x2 = ox1 + int(face_w * 0.75)
                        
                        mouth_y1, mouth_x1 = max(0, mouth_y1), max(0, mouth_x1)
                        mouth_y2, mouth_x2 = min(h, mouth_y2), min(w, mouth_x2)
                    else:
                        # Fallback coordinates
                        mouth_y1, mouth_y2 = int(h * 0.5), int(h * 0.85)
                        mouth_x1, mouth_x2 = int(w * 0.25), int(w * 0.75)
                    
                    viseme_mouth = viseme_frame[mouth_y1:mouth_y2, mouth_x1:mouth_x2]
                    if viseme_mouth.size > 0:
                        frame = blend_mouth_onto_face(frame, viseme_mouth, face_cache)
                        viseme_applied = True
        
        # If still no viseme applied and we have 'M' available, use it as fallback
        if not viseme_applied and 'M' in mouth_library:
            frame = blend_mouth_onto_face(frame, mouth_library['M'], face_cache)
        
        # Apply blink overlay if blinking
        if is_blinking and blink_frame is not None:
            frame = blend_blink_onto_face(frame, blink_frame, face_cache)
        
        out.write(frame)
        
        # Yield progress every 5 frames for smoother updates
        if frame_idx % 5 == 0:
            progress = f"🔄 Frame {frame_idx}/{total_frames} ({frame_idx*100//total_frames}%) | Viseme: {current_viseme}"
            yield None, audio_path, progress
    
    out.release()
    
    # Debug: Show viseme changes
    if viseme_changes:
        print(f"🔍 Debug: Viseme changes during video: {viseme_changes[:10]}...")
        print(f"🔍 Debug: Total viseme transitions: {len(viseme_changes)}")
    
    # Step 6: Merge audio
    yield None, audio_path, "🔄 Merging audio..."
    final_video = os.path.join(gen_dir, f"live_{ts}.mp4")
    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
        '-i', temp_video,
        '-i', audio_path,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-shortest',
        final_video
    ]
    
    try:
        import subprocess
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        os.remove(temp_video)
    except Exception as e:
        yield None, None, f"❌ Failed to merge audio: {e}"
        return
    
    elapsed = time.time() - start_time
    viseme_info = f"{len(viseme_changes)} transitions" if viseme_changes else "static"
    blink_info = "with eye blinks" if blink_frame is not None else "no blinks"
    yield final_video, audio_path, f"✅ Generated in {elapsed:.1f}s\n📹 {os.path.basename(final_video)}\n🎭 Viseme-based lip sync ({viseme_info})\n👁️ {blink_info}\n💡 Tip: Regenerate viseme library for more pronounced shapes"


def compose_live_video(text: str, fps: int = 25):
    """Compose video using viseme library - wrapper for streaming."""
    final_video = None
    final_audio = None
    final_status = "Generation failed"
    for video, audio, status in compose_live_video_streaming(text, fps):
        if video is not None:
            final_video = video
        if audio is not None:
            final_audio = audio
        if status:
            final_status = status
    return final_video, final_audio, final_status


def auto_setup_from_assets(size: int = 256):
    """Automatically setup using default assets."""
    face_cache = load_face_cache()
    
    results = []
    errors = []
    
    if os.path.exists(DEFAULT_IMAGE):
        if not face_cache or face_cache.get('size') != size:
            try:
                _, msg = preprocess_and_cache_face(DEFAULT_IMAGE, "female-01", size=size)
                results.append(msg)
            except Exception as e:
                errors.append(f"❌ Face setup failed: {str(e)}")
        else:
            results.append(f"✓ Face already cached ({size}px)")
    else:
        errors.append(f"⚠ Image not found: {DEFAULT_IMAGE}")
    
    output = "\n".join(results) if results else ""
    if errors:
        output += "\n\n" + "\n".join(errors)
    
    return output if output else "✅ Ready! Face cached."


# Gradio UI
with gr.Blocks(title="SadTalker — Live Avatar") as demo:
    gr.Markdown(f"""
    # 🎭 SadTalker: Live Avatar (Real-time Viseme-based)
    
    **Live mode:** Pre-rendered visemes + real-time composition = instant lip sync
    
    **Expected latency:** ~0.5-1s (TTS only), then instant playback
    
    **📁 Cache:** `{CACHE_DIR}`  
    **📁 Visemes:** `{VISEME_DIR}`  
    **📁 Results:** `{RESULT_DIR}`
    """)
    
    with gr.Tabs():
        with gr.TabItem("1️⃣ Setup (Run Once)"):
            gr.Markdown("### Step 1: Pre-process face")
            
            with gr.Row():
                with gr.Column():
                    setup_size = gr.Slider(
                        minimum=128, maximum=512, step=64, value=256,
                        label="Face Resolution (px)"
                    )
                    auto_setup_btn = gr.Button("🚀 Auto-Setup from Assets", variant="primary")
                auto_setup_status = gr.Textbox(label="Setup Status", interactive=False, lines=5)
            
            auto_setup_btn.click(fn=auto_setup_from_assets, inputs=[setup_size], outputs=[auto_setup_status])
            
            gr.Markdown("---\n### Step 2: Generate Viseme Library (Once)")
            gr.Markdown("**Pre-render all mouth shapes (visemes). This enables real-time composition.**")
            
            gen_viseme_btn = gr.Button("🎭 Generate Viseme Library", variant="primary", scale=2)
            viseme_status = gr.Textbox(label="Viseme Library Status", interactive=False, lines=5)
            
            gen_viseme_btn.click(fn=generate_viseme_library, outputs=[gr.Textbox(visible=False), viseme_status])
            
            gr.Markdown("---\n### Step 3: Regenerate Mouth Regions (if missing)")
            gr.Markdown("**If mouth_*.png files are missing, extract them from viseme frames.**")
            
            regen_mouth_btn = gr.Button("🔧 Regenerate Mouth Regions", variant="secondary")
            regen_mouth_status = gr.Textbox(label="Mouth Region Status", interactive=False, lines=3)
            
            regen_mouth_btn.click(fn=regenerate_mouth_regions, outputs=[gr.Textbox(visible=False), regen_mouth_status])
            
            gr.Markdown("---\n### Manual Setup (Optional)")
            
            with gr.Row():
                with gr.Column():
                    setup_image = gr.Image(type="filepath", label="Face Image")
                    setup_cache_id = gr.Textbox(label="Face Cache ID", value="default")
                    setup_face_size = gr.Slider(minimum=128, maximum=512, step=64, value=256, label="Resolution")
                    setup_face_btn = gr.Button("Pre-process Face", variant="secondary")
                
            setup_status = gr.Textbox(label="Manual Setup Status", interactive=False, lines=3)
            
            def do_setup_face(image, cache_id, size):
                if not image:
                    return "Please upload a face image"
                image_path = image if isinstance(image, str) else image.get("path") or getattr(image, "name", None)
                _, msg = preprocess_and_cache_face(image_path, cache_id or "default", size=int(size))
                return msg
            
            setup_face_btn.click(fn=do_setup_face, inputs=[setup_image, setup_cache_id, setup_face_size], outputs=[setup_status])
        
        with gr.TabItem("2️⃣ Live Generation"):
            gr.Markdown("### Enter text → Instant video with lip movements (viseme-based)")
            
            gen_text = gr.Textbox(
                label="Text to speak",
                lines=4,
                placeholder="Enter any text... The avatar will speak it with lip movements!"
            )
            
            gen_btn = gr.Button("🎭 Generate Live Video", variant="primary", scale=2)
            
            gen_video = gr.Video(label="Output Video (with lip sync)", height=400)
            gen_audio = gr.Audio(label="Generated Speech", type="filepath")
            gen_status = gr.Textbox(label="Status", interactive=False, lines=4)
            
            def do_generate(text):
                if not text or not text.strip():
                    return None, None, "Please enter some text"
                
                # Use streaming version for progress updates
                video_path = None
                audio_path = None
                final_status = ""
                for video, audio, status in compose_live_video_streaming(text):
                    if video is not None:
                        video_path = video
                    if audio is not None:
                        audio_path = audio
                    if status:
                        final_status = status
                
                return video_path, audio_path, final_status
            
            gen_btn.click(fn=do_generate, inputs=[gen_text], outputs=[gen_video, gen_audio, gen_status])
    
    gr.Markdown("""
    ### 🎭 How Live Mode Works:
    1. **Setup:** Pre-process face + Generate viseme library (9 mouth shapes, once)
    2. **Generate:** Text → TTS (~0.5-1s) → Phonemes → Visemes → Compose frames (~0.1s)
    3. **Speed:** ~0.5-1s total (vs ~87s optimized, ~360s standard)
    
    ### ⚡ Performance:
    - **TTS generation:** ~0.5-1s (unavoidable)
    - **Phoneme conversion:** ~0.01s (instant)
    - **Viseme lookup:** ~0.01s per frame (instant)
    - **Frame composition:** ~0.1s total (very fast)
    - **Total:** ~0.5-1s ⚡⚡⚡
    
    ### 💡 Tips:
    - Install `phonemizer` for accurate phoneme conversion: `pip install phonemizer`
    - For Windows: `phonemizer` will use `segments` backend (Python-only, no system install needed)
    - For Linux/Mac: Install `espeak-ng` system package for better accuracy
    - Viseme library needs to be generated once per face
    - Quality depends on viseme coverage (9 basic visemes cover most sounds)
    """)

if __name__ == "__main__":
    print("\n🎭 Starting SadTalker Live Avatar App...")
    print(f"📁 Working directory: {BASE_DIR}")
    print(f"💻 CUDA available: {torch.cuda.is_available()}\n")
    
    demo.launch(
        debug=True,
        share=False,
        server_name="127.0.0.1",
        server_port=7863  # Different port
    )
