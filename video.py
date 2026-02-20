"""
video.py — Real-time lip-sync overlay on an existing video
============================================================

What this does
--------------
1. Plays video01.mp4 frame-by-frame in a cv2 window.
2. At the same time, TTS audio plays in the background.
3. On every frame, the correct viseme mouth shape (pre-generated during --setup)
   is blended onto the person's mouth region in that frame.

This is NOT a video generator — it just drives the mouth of the person
already in the video in sync with whatever text you give it.

Usage
-----
# One-time setup (face + viseme library, same person, takes ~5 min):
  python video.py --setup

# Real-time lip-sync playback (opens a window):
  python video.py --text "Hello, how are you today?"

# Save the result too (optional):
  python video.py --text "Hello, how are you today?" --save out.mp4

# Gradio UI:
  python video.py --ui
"""

import os, sys, pickle, asyncio, argparse, random, subprocess, time, threading
from datetime import datetime
import cv2
import numpy as np
import torch
import edge_tts
from pydub import AudioSegment

# ── numpy 2.x compat ─────────────────────────────────────────────────────────
if not hasattr(np, 'float'):  np.float = float
if not hasattr(np, 'int'):    np.int   = int

# ── Auto-detect BASE_DIR ──────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
BASE_DIR = None
for _p in [script_dir, os.path.join(script_dir, "SadTalker"), os.path.dirname(script_dir)]:
    if os.path.exists(os.path.join(_p, "inference.py")):
        BASE_DIR = _p; break
if BASE_DIR is None:
    print("❌ Could not find SadTalker directory"); sys.exit(1)
sys.path.insert(0, BASE_DIR)

# ── Paths ─────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR      = os.path.join(BASE_DIR, "checkpoints")
CACHE_DIR           = os.path.join(BASE_DIR, "cache")
RESULT_DIR          = os.path.join(BASE_DIR, "results")
VISEME_DIR          = os.path.join(CACHE_DIR, "visemes")
VIDEO_FACE_CACHE    = os.path.join(CACHE_DIR, "video_face_cache.pkl")
VIDEO_VISEME_LIB    = os.path.join(CACHE_DIR, "video_viseme_library.pkl")
for _d in [CACHE_DIR, RESULT_DIR, VISEME_DIR]:
    os.makedirs(_d, exist_ok=True)

print(f"📁 BASE_DIR : {BASE_DIR}")
print(f"🎬 video.py : Real-time video lip-sync")

# ── Viseme / phoneme constants ────────────────────────────────────────────────
VISEME_TYPES = ['Sil','Ax','Aa','Ao','Eh','Er','Iy','W','Oh','Ow',
                'Oy','Ay','H','R','L','S','Sh','Th','F','T','K','M']
BLINK_VISEME = 'BLINK'

PHONEME_TO_VISEME = {
    'SIL':'Sil','SP':'Sil','':'Sil',
    'AH':'Ax','AX':'Ax','AX-H':'Ax',
    'AA':'Aa','AA0':'Aa','AA1':'Aa','AA2':'Aa',
    'AO':'Ao','AO0':'Ao','AO1':'Ao','AO2':'Ao',
    'EH':'Eh','EH0':'Eh','EH1':'Eh','EH2':'Eh',
    'ER':'Er','ER0':'Er','ER1':'Er','ER2':'Er','AXR':'Er',
    'IY':'Iy','IY0':'Iy','IY1':'Iy','IY2':'Iy',
    'W':'W','UW':'W','UW0':'W','UW1':'W','UW2':'W',
    'OW':'Oh','OW0':'Oh','OW1':'Oh','OW2':'Oh',
    'AW':'Ow','AW0':'Ow','AW1':'Ow','AW2':'Ow',
    'OY':'Oy','OY0':'Oy','OY1':'Oy','OY2':'Oy',
    'AY':'Ay','AY0':'Ay','AY1':'Ay','AY2':'Ay',
    'HH':'H',
    'R':'R','L':'L','S':'S','Z':'S',
    'SH':'Sh','ZH':'Sh','CH':'Sh','JH':'Sh',
    'DH':'Th','F':'F','V':'F',
    'T':'T','D':'T','N':'T','TH':'T',
    'K':'K','G':'K','NG':'K',
    'M':'M','B':'M','P':'M',
    'IH':'Iy','IH0':'Iy','IH1':'Iy','IH2':'Iy',
    'EY':'Eh','AE':'Eh','UH':'W','Y':'Iy',
}
for _ip, _v in [('æ','Eh'),('ə','Ax'),('ʌ','Ax'),('ɑ','Aa'),('ɔ','Ao'),
                ('ɛ','Eh'),('ʊ','Eh'),('ɝ','Er'),('iː','Iy'),('uː','W'),
                ('oʊ','Oh'),('aʊ','Ow'),('ɔɪ','Oy'),('aɪ','Ay'),
                ('h','H'),('r','R'),('l','L'),('s','S'),('z','S'),
                ('ʃ','Sh'),('ʒ','Sh'),('θ','T'),('ð','Th'),('f','F'),
                ('v','F'),('t','T'),('d','T'),('n','T'),('k','K'),
                ('g','K'),('ŋ','K'),('m','M'),('b','M'),('p','M'),
                ('w','W'),('j','Iy')]:
    PHONEME_TO_VISEME[_ip] = _v

# Blink constants (breathing removed — source video has natural movement)
BLINK_DURATION_FRAMES   = 5
BLINK_CURVE             = [0.2, 0.7, 1.0, 0.7, 0.2]
BLINK_INTERVAL_MIN      = 48
BLINK_INTERVAL_MAX      = 175
BLINK_JITTER            = 12
EYE_TOP, EYE_BOT        = 0.20, 0.48
EYE_LEFT, EYE_RIGHT     = 0.08, 0.92
VISEME_SMOOTH_FRAMES    = 1   # low = crisp mouths; was 4 (caused blur)


# ═══════════════════════════════════════════════════════════════════════════════
#  AUDIO PLAYBACK (pygame preferred, ffplay fallback)
# ═══════════════════════════════════════════════════════════════════════════════

def play_audio_background(wav_path: str):
    """Start audio playback in a daemon thread. Returns the thread."""
    def _play():
        try:
            import pygame
            pygame.mixer.init()
            pygame.mixer.music.load(wav_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.01)
        except Exception:
            # fallback: ffplay (silent if not installed)
            try:
                subprocess.Popen(
                    ['ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet', wav_path],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass
    t = threading.Thread(target=_play, daemon=True)
    t.start()
    return t


# ═══════════════════════════════════════════════════════════════════════════════
#  PHONEME / VISEME HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def text_to_phonemes_simple(text: str):
    text = text.upper().strip()
    if not text: return []
    word_pat = {
        'HI':['HH','AY'],'HELLO':['HH','EH','L','OW'],
        'THE':['DH','AH'],'A':['AH'],'AN':['AE','N'],
        'IS':['IH','Z'],'ARE':['AA','R'],'WAS':['W','AA','Z'],
        'TO':['T','UW'],'OF':['AH','V'],'AND':['AE','N','D'],
        'IN':['IH','N'],'ON':['AA','N'],'AT':['AE','T'],
        'IT':['IH','T'],'THIS':['DH','IH','S'],'THAT':['DH','AE','T'],
        'WITH':['W','IH','TH'],'FOR':['F','AO','R'],
        'YOU':['Y','UW'],'WHAT':['W','AA','T'],
        'YES':['Y','EH','S'],'NO':['N','OW'],
    }
    def cph(c, p=None, n=None):
        nc = n or ''
        if c==' ': return ['SP']
        if c not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ': return []
        if c=='A': return ['EY'] if nc in 'EIY' else ['AE']
        if c=='E': return ['IY'] if nc=='E' else ([] if p=='I' else ['EH'])
        if c=='I': return ['AY'] if nc in 'EGH' else ['IH']
        if c=='O': return ['UW'] if nc=='O' else (['OW'] if nc=='W' else ['AO'])
        if c=='U': return ['W'] if p=='Q' else ['UH']
        m={'B':'B','C':('CH' if nc=='H' else ('S' if nc in 'EIY' else 'K')),
           'D':'D','F':'F','G':('JH' if nc in 'EIY' else 'G'),'H':'HH',
           'J':'JH','K':'K','L':'L','M':'M','N':'N','P':'P',
           'R':'R','S':('SH' if nc=='H' else 'S'),
           'T':('TH' if nc=='H' else 'T'),'V':'V','W':'W','Y':'Y','Z':'Z'}
        v = m.get(c)
        if v: return [v]
        return ['K','W'] if c=='Q' else (['K','S'] if c=='X' else ['AH'])
    phonemes = []
    for word in text.split():
        if word in word_pat: phonemes.extend(word_pat[word])
        else:
            for i,ch in enumerate(word):
                phonemes.extend(cph(ch, word[i-1] if i>0 else None,
                                       word[i+1] if i<len(word)-1 else None))
        phonemes.append('SP')
    if phonemes and phonemes[-1]=='SP': phonemes.pop()
    return phonemes or ['AH']


def text_to_phonemes(text: str):
    try:
        from phonemizer import phonemize
        try:
            from phonemizer.backend import EspeakBackend
            return phonemize(text, backend=EspeakBackend('en-us'), separator=' ', strip=True).split()
        except Exception:
            pass
    except Exception:
        pass
    return text_to_phonemes_simple(text)


def phonemes_to_visemes(phonemes):
    out = []
    for ph in phonemes:
        key = ph.upper() if isinstance(ph,str) else ph
        out.append(PHONEME_TO_VISEME.get(key) or PHONEME_TO_VISEME.get(ph) or 'Ax')
    return out


async def _tts_async(text, voice, out_path):
    mp3 = out_path.replace('.wav','.mp3')
    await edge_tts.Communicate(text, voice).save(mp3)
    AudioSegment.from_mp3(mp3).export(out_path, format='wav')
    if os.path.exists(mp3): os.remove(mp3)


def detect_silence_periods(wav_path, thresh=-35.0, min_dur=0.15):
    try:
        audio = AudioSegment.from_wav(wav_path)
        if audio.channels>1: audio=audio.set_channels(1)
        periods, in_sil, sil_start = [], False, 0.0
        for i, chunk in enumerate(audio[::100]):
            t=i*0.1
            if chunk.dBFS < thresh:
                if not in_sil: in_sil,sil_start=True,t
            else:
                if in_sil:
                    if t-sil_start>=min_dur: periods.append((sil_start,t))
                    in_sil=False
        if in_sil:
            d=len(audio)/1000-sil_start
            if d>=min_dur: periods.append((sil_start,len(audio)/1000))
        return periods
    except: return []


# ═══════════════════════════════════════════════════════════════════════════════
#  FACE HELPERS — crop_info-based bbox extraction
# ═══════════════════════════════════════════════════════════════════════════════

def face_bbox_from_cache(face_cache, w, h):
    """Return (ox1,oy1,ox2,oy2) face bbox in image coords."""
    ci = face_cache.get('crop_info')
    if ci and len(ci)==3:
        _,(clx,cly,crx,cry),(lx,ly,rx,ry) = ci
        ox1=max(0,int(clx+lx)); oy1=max(0,int(cly+ly))
        ox2=min(w,int(clx+rx)); oy2=min(h,int(cly+ry))
        return ox1,oy1,ox2,oy2
    # fallback: whole image
    return 0, int(h*0.1), w, h


def mouth_bbox_from_face(ox1, oy1, ox2, oy2, w, h, face_cache=None):
    # ── prefer saved landmark bbox (set during setup) ──────────────────────
    if face_cache is not None:
        frac = face_cache.get('mouth_lm_frac')
        if frac is not None:
            fx1, fy1, fx2, fy2 = frac
            return (max(0, int(fx1 * w)), max(0, int(fy1 * h)),
                    min(w, int(fx2 * w)), min(h, int(fy2 * h)))
    # ── fallback: ratio from face bbox ─────────────────────────────────────
    fh, fw = oy2-oy1, ox2-ox1
    my1 = max(0, oy1 + int(fh * 0.60));  my2 = min(h, oy1 + int(fh * 0.84))
    mx1 = max(0, ox1 + int(fw * 0.25));  mx2 = min(w, ox1 + int(fw * 0.75))
    return mx1, my1, mx2, my2


def detect_and_save_mouth_lm(ref_img_path, face_cache, w, h):
    """
    Use face_alignment (SadTalker dep) to detect 68-point landmarks on the
    reference frame and store the mouth bbox as normalised fractions.
    Returns True on success, False on fallback.
    """
    try:
        import face_alignment as _fa
        fa = _fa.FaceAlignment(_fa.LandmarksType.TWO_D, flip_input=False, device='cpu')
        img = cv2.imread(ref_img_path)
        if img is None: raise ValueError("Cannot read reference image")
        h_img, w_img = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        preds = fa.get_landmarks(rgb)
        del fa
        if not preds or len(preds) == 0:
            raise ValueError("No landmarks found")
        pts = preds[0]                  # (68, 2) — x, y pixel coords
        mouth = pts[48:68]             # outer + inner lip points
        xs, ys = mouth[:, 0], mouth[:, 1]
        pad_x = (xs.max() - xs.min()) * 0.12
        pad_y = (ys.max() - ys.min()) * 0.18
        mx1 = max(0, int(xs.min() - pad_x))
        mx2 = min(w_img, int(xs.max() + pad_x))
        my1 = max(0, int(ys.min() - pad_y))
        my2 = min(h_img, int(ys.max() + pad_y))
        # Store as normalised fractions (resolution-independent)
        face_cache['mouth_lm_frac'] = (mx1/w_img, my1/h_img, mx2/w_img, my2/h_img)
        print(f"✅ Mouth landmarks: px=({mx1},{my1},{mx2},{my2})  "
              f"frac=({mx1/w_img:.3f},{my1/h_img:.3f},{mx2/w_img:.3f},{my2/h_img:.3f})")
        return True
    except Exception as e:
        print(f"⚠ Landmark detection failed ({e}) — ratio-based fallback used")
        return False


def eye_bbox_from_face(ox1, oy1, ox2, oy2, w, h):
    fh,fw = oy2-oy1, ox2-ox1
    ey1=max(0, oy1+int(fh*EYE_TOP));  ey2=min(h, oy1+int(fh*EYE_BOT))
    ex1=max(0, ox1+int(fw*EYE_LEFT)); ex2=min(w, ox1+int(fw*EYE_RIGHT))
    return ex1,ey1,ex2,ey2


# ═══════════════════════════════════════════════════════════════════════════════
#  BLENDING — blend a mouth/eye patch onto a frame
# ═══════════════════════════════════════════════════════════════════════════════

def _ellipse_mask(h, w, softness=0.35):
    """Elliptical soft mask: 1.0 at centre, fades to 0 at edge."""
    cy, cx = (h-1) / 2.0, (w-1) / 2.0
    y = np.linspace(0, h-1, h)
    x = np.linspace(0, w-1, w)
    xx, yy = np.meshgrid(x, y)
    # Normalised ellipse distance (0=centre, 1=edge)
    dist = np.sqrt(((xx - cx) / max(cx, 1))**2 + ((yy - cy) / max(cy, 1))**2)
    # Smoothstep: 0 outside, 1 inside, soft transition in 'softness' band
    inner = 1.0 - softness
    mask = np.clip((1.0 - dist) / softness, 0.0, 1.0)
    mask = mask * mask * (3 - 2 * mask)          # smoothstep
    return mask.astype(np.float32)


def blend_patch(frame, patch, x1, y1, x2, y2, strength=0.70):
    """Blend patch into frame[y1:y2, x1:x2] using an elliptical soft mask."""
    rh, rw = y2-y1, x2-x1
    if rh <= 0 or rw <= 0 or patch is None or patch.size == 0:
        return frame
    patch_r = cv2.resize(patch, (rw, rh))
    mask    = _ellipse_mask(rh, rw, softness=0.40)
    mask_3d = np.stack([mask]*3, axis=2) * strength
    result  = frame.copy()
    region  = result[y1:y2, x1:x2].astype(np.float32)
    blended = (region * (1 - mask_3d) + patch_r.astype(np.float32) * mask_3d).astype(np.uint8)
    result[y1:y2, x1:x2] = blended
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  ONE-TIME SETUP  (extract frame + build viseme library)
# ═══════════════════════════════════════════════════════════════════════════════

def setup(video_path: str, frame_idx: int = -1, size: int = 256):
    """
    Extract reference frame from video, pre-process face, build viseme library.
    Must be run once before realtime lip-sync.
    """
    # ── 1. extract reference frame ───────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open: {video_path}"); return False
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target = frame_idx if frame_idx>=0 else total//2
    target = max(0, min(target, total-1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f"❌ Could not read frame {target}"); return False

    vid_name = os.path.splitext(os.path.basename(video_path))[0]
    ref_path = os.path.join(CACHE_DIR, f"{vid_name}_ref_frame.png")
    cv2.imwrite(ref_path, frame)
    print(f"✅ Reference frame saved: {ref_path}")

    # ── 2. face pre-processing (SadTalker CropAndExtract) ────────────────────
    print("🔄 Pre-processing face (once)…")
    from src.utils.preprocess import CropAndExtract
    from src.utils.init_path import init_path
    device = "cuda" if torch.cuda.is_available() else "cpu"
    paths  = init_path(CHECKPOINT_DIR, os.path.join(BASE_DIR,'src/config'), size, False, 'full')
    model  = CropAndExtract(paths, device)
    fdir   = os.path.join(CACHE_DIR, "face_video_ref")
    os.makedirs(fdir, exist_ok=True)
    coeff, crop_pic, crop_info = model.generate(ref_path, fdir, 'full',
                                                source_image_flag=True, pic_size=size)
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    if coeff is None:
        print("❌ Face detection failed — try a different frame"); return False

    face_cache = {
        'first_coeff_path': coeff,
        'crop_pic_path':    crop_pic,
        'crop_info':        crop_info,
        'image_path':       ref_path,
        'size':             size,
    }

    # ── 2b. precise mouth landmark detection ─────────────────────────────────
    h_ref, w_ref = frame.shape[:2]
    detect_and_save_mouth_lm(ref_path, face_cache, w_ref, h_ref)

    with open(VIDEO_FACE_CACHE, 'wb') as f:
        pickle.dump(face_cache, f)
    print("✅ Face cached")

    # ── 3. viseme library ─────────────────────────────────────────────────────
    print("🎭 Building viseme library (takes a few minutes)…")
    _build_viseme_library(face_cache, size, device)
    return True


def _build_viseme_library(face_cache, size, device):
    from src.utils.init_path import init_path
    from src.test_audio2coeff import Audio2Coeff
    from src.facerender.animate import AnimateFromCoeff
    from src.generate_batch import get_data
    from src.generate_facerender_batch import get_facerender_data

    paths  = init_path(CHECKPOINT_DIR, os.path.join(BASE_DIR,'src/config'), size, False, 'full')
    a2c    = Audio2Coeff(paths, device)
    afc    = AnimateFromCoeff(paths, device)

    audio_texts = {
        'Sil':'mmm','Ax':'a banana','Aa':'ah ah ah','Ao':'aw aw aw',
        'Eh':'eh eh eh','Er':'her bird','Iy':'ee ee ee','W':'woo woo woo',
        'Oh':'oh oh oh','Ow':'how now','Oy':'toy boy','Ay':'my eye',
        'H':'ha ha ha','R':'rrr rrr','L':'la la la','S':'sss sss',
        'Sh':'shh shh','Th':'the the','F':'fff fff','T':'tah tah',
        'K':'kuh kuh','M':'mmm mmm',
    }

    library = {}

    def _gen(vtype, atext, is_blink=False):
        tmp = os.path.join(VISEME_DIR, f"tmp_{vtype}.wav")
        asyncio.run(_tts_async(atext, "en-US-JennyNeural", tmp))
        gdir = os.path.join(VISEME_DIR, f"vid_{vtype}")
        os.makedirs(gdir, exist_ok=True)
        batch = get_data(face_cache['first_coeff_path'], tmp, device,
                         ref_eyeblink_coeff_path=None, still=True, use_blink=True)
        cp = a2c.generate(batch, gdir, pose_style=0, ref_pose_coeff_path=None)

        # Use reference frame's natural pose (None = SadTalker reads it from
        # first_coeff_path). still_mode=True keeps head stable without forcing
        # absolute 0-degrees which would mismatch the video's head tilt.
        data = get_facerender_data(cp, face_cache['crop_pic_path'],
                                   face_cache['first_coeff_path'], tmp,
                                   batch_size=2,
                                   input_yaw_list=None, input_pitch_list=None, input_roll_list=None,
                                   expression_scale=1.0, still_mode=True,
                                   preprocess='full', size=size)
        res = afc.generate(data, gdir, face_cache['image_path'], face_cache['crop_info'],
                           enhancer=None, background_enhancer=None,
                           preprocess='full', img_size=size)
        if not res or not os.path.exists(res): return

        cap = cv2.VideoCapture(res)
        fc  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # For blink: use mid→end frame; for viseme: use 1/3→2/3 frame
        start = fc//2 if is_blink else max(1, fc//3)
        end   = min(fc-1, fc//2+fc//3) if is_blink else min(fc-1, fc*2//3)
        best  = None
        for fi in range(start, end):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frm = cap.read()
            if ok: best=frm; break
        if best is None and fc>0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fc//2)
            _, best = cap.read()
        cap.release()
        if best is None: return

        # Save full frame
        fpath = os.path.join(VISEME_DIR, f"viseme_vid_{vtype}.png")
        cv2.imwrite(fpath, best)
        library[vtype] = fpath

        # Extract and save mouth patch using crop_info
        bh, bw = best.shape[:2]
        ox1,oy1,ox2,oy2 = face_bbox_from_cache(face_cache, bw, bh)
        mx1,my1,mx2,my2 = mouth_bbox_from_face(ox1,oy1,ox2,oy2, bw, bh, face_cache)
        mouth = best[my1:my2, mx1:mx2]
        if mouth.size>0:
            cv2.imwrite(os.path.join(VISEME_DIR, f"mouth_vid_{vtype}.png"), mouth)

        # Extract and save eye patch (for blink)
        if is_blink:
            ex1,ey1,ex2,ey2 = eye_bbox_from_face(ox1,oy1,ox2,oy2, bw, bh)
            eye = best[ey1:ey2, ex1:ex2]
            if eye.size>0:
                cv2.imwrite(os.path.join(VISEME_DIR, "eye_blink.png"), eye)

    for vt in VISEME_TYPES:
        print(f"   {vt}…")
        _gen(vt, audio_texts.get(vt,'ah'))
    print("   BLINK…")
    _gen(BLINK_VISEME, 'mm', is_blink=True)

    with open(VIDEO_VISEME_LIB, 'wb') as f:
        pickle.dump(library, f)

    del a2c, afc
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(f"✅ Viseme library ready ({len(library)} entries)")


# ═══════════════════════════════════════════════════════════════════════════════
#  LOAD CACHES
# ═══════════════════════════════════════════════════════════════════════════════

def load_face_cache():
    if os.path.exists(VIDEO_FACE_CACHE):
        with open(VIDEO_FACE_CACHE,'rb') as f: return pickle.load(f)
    return None

def load_viseme_library():
    if os.path.exists(VIDEO_VISEME_LIB):
        with open(VIDEO_VISEME_LIB,'rb') as f: return pickle.load(f)
    return None

def load_mouth_library(face_cache, ref_w, ref_h):
    """Load all mouth_vid_*.png patches into memory for fast per-frame lookup."""
    mouth_lib = {}
    for vt in VISEME_TYPES:
        p = os.path.join(VISEME_DIR, f"mouth_vid_{vt}.png")
        if os.path.exists(p):
            img = cv2.imread(p)
            if img is not None:
                mouth_lib[vt] = img
    # blink eye patch
    bp = os.path.join(VISEME_DIR, "eye_blink.png")
    if os.path.exists(bp):
        mouth_lib[BLINK_VISEME] = cv2.imread(bp)
    return mouth_lib


# ═══════════════════════════════════════════════════════════════════════════════
#  BLINK SCHEDULE
# ═══════════════════════════════════════════════════════════════════════════════

def make_blink_schedule(n_frames, fps, silence_periods=None):
    ratio = np.zeros(n_frames, dtype=np.float32)
    if n_frames<=30: return ratio
    if silence_periods:
        for s,e in silence_periods:
            sf,ef = int(s*fps),int(e*fps)
            if ef-sf>=BLINK_DURATION_FRAMES+2 and sf<n_frames:
                ms=min(sf+(ef-sf-BLINK_DURATION_FRAMES), n_frames-BLINK_DURATION_FRAMES-1)
                if ms>=sf:
                    bs=random.randint(sf,max(sf,ms))
                    for i,v in enumerate(BLINK_CURVE):
                        idx=bs+i
                        if 0<=idx<n_frames: ratio[idx]=max(ratio[idx],v)
    imin=min(BLINK_INTERVAL_MIN,n_frames//4)
    imax=min(BLINK_INTERVAL_MAX,n_frames//2)
    if imax<=imin: imax=imin+20
    nxt=random.randint(imin//2, min(imax,n_frames-BLINK_DURATION_FRAMES-5))
    while nxt+BLINK_DURATION_FRAMES<n_frames:
        for i,v in enumerate(BLINK_CURVE):
            idx=nxt+i
            if 0<=idx<n_frames: ratio[idx]=max(ratio[idx],v)
        nxt+=BLINK_DURATION_FRAMES+max(10,random.randint(imin,imax)+random.randint(-BLINK_JITTER,BLINK_JITTER))
    return ratio


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN: REAL-TIME LIP-SYNC ON VIDEO
# ═══════════════════════════════════════════════════════════════════════════════

def _read_next_frame(cap, total_vid):
    """Read next frame, looping back seamlessly when the video ends."""
    ok, frame = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
    return ok, frame


def _idle_loop(cap, total_vid, fps, window_name, duration_secs, writer=None):
    """
    Play the raw video (idle / natural movement) for *duration_secs*.
    Returns False if the user pressed Q/Esc during idle.
    """
    n = int(duration_secs * fps)
    t0 = time.perf_counter()
    for fi in range(n):
        ok, frame = _read_next_frame(cap, total_vid)
        if not ok: break
        cv2.putText(frame, "Idle", (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (120,120,120), 1, cv2.LINE_AA)
        cv2.imshow(window_name, frame)
        if writer: writer.write(frame)
        # pace to fps
        elapsed = time.perf_counter() - t0
        sleep   = (fi+1)/fps - elapsed
        if sleep > 0: time.sleep(sleep)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            return False
    return True


def realtime_lipsync(
        video_path: str,
        text: str,
        voice: str = "en-US-JennyNeural",
        fps_override: int = 0,
        save_path: str = "",
        idle_before: float = 0.4,
        idle_after:  float = 0.6,
        window_name: str = "Live Lip-Sync  [Q to quit]"):
    """
    Play video_path in a cv2 window with real-time viseme-driven mouth blending.

    Phases
    ------
    1. IDLE  (idle_before seconds) - raw video loops, natural movement
    2. SPEAK - same loop but mouth patched with visemes + blink overlay
    3. IDLE  (idle_after  seconds) - back to raw video loop

    Breathing is NOT added here — the source video already has it.
    """
    # ── load caches ───────────────────────────────────────────────────────────
    face_cache = load_face_cache()
    if not face_cache:
        print("❌ No face cache. Run: python video.py --setup"); return

    viseme_lib = load_viseme_library()
    if not viseme_lib:
        print("❌ No viseme library. Run: python video.py --setup"); return

    # ── open video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open: {video_path}"); return

    vid_fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    fps       = fps_override if fps_override > 0 else vid_fps
    total_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"📹 Video: {total_vid} frames @ {fps:.1f} fps  ({vid_w}×{vid_h})")

    # ── TTS ───────────────────────────────────────────────────────────────────
    print("🔄 Generating speech…")
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    wav = os.path.join(CACHE_DIR, f"tts_{ts}.wav")
    asyncio.run(_tts_async(text.strip(), voice, wav))
    print(f"✅ Audio: {wav}")

    # ── phonemes → viseme sequence ────────────────────────────────────────────
    phonemes     = text_to_phonemes(text)
    visemes_raw  = phonemes_to_visemes(phonemes)
    visemes_merged = []
    for v in visemes_raw:
        v = 'M' if v == 'Sil' else v
        if not visemes_merged or visemes_merged[-1] != v:
            visemes_merged.append(v)
    if not visemes_merged: visemes_merged = ['M']

    audio_dur   = len(AudioSegment.from_wav(wav)) / 1000.0
    tts_frames  = int(audio_dur * fps)
    silence_per = detect_silence_periods(wav)
    fpv = max(3, min(12, tts_frames // max(len(visemes_merged), 1)))
    print(f"🎭 {len(phonemes)} phonemes → {len(visemes_merged)} visemes  ({fpv} f/v)")

    # ── load patches ──────────────────────────────────────────────────────────
    mouth_lib = load_mouth_library(face_cache, vid_w, vid_h)
    print(f"✓ Mouth library: {len(mouth_lib)} patches")

    ox1,oy1,ox2,oy2 = face_bbox_from_cache(face_cache, vid_w, vid_h)
    mx1,my1,mx2,my2 = mouth_bbox_from_face(ox1,oy1,ox2,oy2, vid_w, vid_h)
    ex1,ey1,ex2,ey2 = eye_bbox_from_face(ox1,oy1,ox2,oy2, vid_w, vid_h)
    blink_ratio = make_blink_schedule(tts_frames, fps, silence_per)

    # ── optional writer ───────────────────────────────────────────────────────
    writer = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(save_path.replace('.mp4','_temp.mp4'),
                                 fourcc, fps, (vid_w, vid_h))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(vid_w*2, 960), min(vid_h*2, 720))

    # ══════════════════════════════════════════════════════════════════════════
    #  PHASE 1 — IDLE (before speech): raw video loops naturally
    # ══════════════════════════════════════════════════════════════════════════
    print(f"💤 Idle  ({idle_before:.1f}s)…")
    if not _idle_loop(cap, total_vid, fps, window_name, idle_before, writer):
        cap.release(); cv2.destroyAllWindows(); return

    # ══════════════════════════════════════════════════════════════════════════
    #  PHASE 2 — SPEAK: loop video + blend viseme mouths
    # ══════════════════════════════════════════════════════════════════════════
    print("🔊 Speaking…")
    audio_thread  = play_audio_background(wav)
    t_start       = time.perf_counter()
    last_viseme   = None
    prev_viseme   = None
    smooth_count  = 0
    tts_fi        = 0
    global_fi     = 0

    while True:
        ok, frame = _read_next_frame(cap, total_vid)
        if not ok: break

        t_now = time.perf_counter() - t_start

        # ── viseme selection ──────────────────────────────────────────────────
        if t_now <= audio_dur:
            is_silent = any(s <= t_now <= e for s, e in silence_per)
            current_viseme = 'M' if is_silent else \
                visemes_merged[min(int(t_now*fps)//fpv, len(visemes_merged)-1)]
        else:
            current_viseme = 'M'

        if current_viseme != last_viseme:
            prev_viseme  = last_viseme
            smooth_count = VISEME_SMOOTH_FRAMES
            last_viseme  = current_viseme

        # ── mouth patch ───────────────────────────────────────────────────────
        curr_mouth = mouth_lib.get(current_viseme)
        if curr_mouth is None:
            vp = viseme_lib.get(current_viseme) or viseme_lib.get('Aa')
            if vp and os.path.exists(vp):
                vf = cv2.imread(vp)
                if vf is not None:
                    curr_mouth = cv2.resize(vf, (vid_w, vid_h))[my1:my2, mx1:mx2]

        # ── smoothstep blend between visemes ──────────────────────────────────
        if smooth_count > 0 and prev_viseme is not None:
            pm = mouth_lib.get(prev_viseme)
            if pm is None: pm = curr_mouth
            if pm is not None and curr_mouth is not None:
                if pm.shape != curr_mouth.shape:
                    pm = cv2.resize(pm, (curr_mouth.shape[1], curr_mouth.shape[0]))
                a = 1.0 - smooth_count/max(VISEME_SMOOTH_FRAMES,1)
                a = a*a*(3-2*a)
                blend_mouth = (pm.astype(np.float32)*(1-a) +
                               curr_mouth.astype(np.float32)*a).astype(np.uint8)
            else:
                blend_mouth = curr_mouth
            smooth_count = max(0, smooth_count-1)
        else:
            blend_mouth = curr_mouth

        # ── overlay mouth ─────────────────────────────────────────────────────
        if blend_mouth is not None:
            frame = blend_patch(frame, blend_mouth, mx1,my1,mx2,my2, strength=0.70)

        # ── blink overlay ─────────────────────────────────────────────────────
        if tts_fi < len(blink_ratio) and blink_ratio[tts_fi] > 0:
            be = mouth_lib.get(BLINK_VISEME)
            if be is not None:
                frame = blend_patch(frame, be, ex1,ey1,ex2,ey2,
                                    strength=blink_ratio[tts_fi])

        # ── HUD ───────────────────────────────────────────────────────────────
        cv2.putText(frame, f"{current_viseme}  {t_now:.1f}s",
                    (10,24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,220,80), 1, cv2.LINE_AA)

        cv2.imshow(window_name, frame)
        if writer: writer.write(frame)

        # ── frame timing ──────────────────────────────────────────────────────
        global_fi += 1
        tts_fi    += 1
        sleep = global_fi/fps - (time.perf_counter()-t_start)
        if sleep > 0: time.sleep(sleep)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            print("🛑 User quit")
            cap.release(); cv2.destroyAllWindows()
            return

        if t_now > audio_dur + 0.3:
            break

    # ══════════════════════════════════════════════════════════════════════════
    #  PHASE 3 — IDLE FOREVER: raw video loops until user presses Q
    # ══════════════════════════════════════════════════════════════════════════
    print("💤 Idling…  (press Q in window to quit)")
    t_idle = time.perf_counter()
    idle_fi = 0
    while True:
        ok, frame = _read_next_frame(cap, total_vid)
        if not ok: break
        cv2.putText(frame, "Idle  [Q to quit]", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120,120,120), 1, cv2.LINE_AA)
        cv2.imshow(window_name, frame)
        if writer: writer.write(frame)
        idle_fi += 1
        sleep = idle_fi/fps - (time.perf_counter() - t_idle)
        if sleep > 0: time.sleep(sleep)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            print("🛑 Quit")
            break

    cap.release()
    cv2.destroyAllWindows()

    # ── finalize save ─────────────────────────────────────────────────────────
    if writer:
        writer.release()
        cmd = ['ffmpeg','-y','-hide_banner','-loglevel','error',
               '-i', save_path.replace('.mp4','_temp.mp4'),
               '-i', wav, '-c:v','copy','-c:a','aac','-shortest', save_path]
        try:
            subprocess.run(cmd, check=True)
            os.remove(save_path.replace('.mp4','_temp.mp4'))
            print(f"✅ Saved: {save_path}")
        except Exception as e:
            print(f"⚠ Merge failed: {e}")

    print("✅ Done")


# ═══════════════════════════════════════════════════════════════════════════════
#  AVATAR THREAD — persistent loop that streams frames into Gradio
# ═══════════════════════════════════════════════════════════════════════════════

import queue as _queue

_current_frame = None            # latest composited frame (numpy BGR)
_text_queue    = _queue.Queue()  # UI → TTS worker: (text, voice)
_audio_ready   = _queue.Queue()  # TTS worker → avatar thread: pre-computed payload
_avatar_active = False


def _tts_worker():
    """
    Separate daemon thread: pulls (text, voice) from _text_queue,
    generates TTS + pre-computes visemes, then pushes a ready payload
    into _audio_ready so the avatar thread never blocks.
    """
    while True:
        try:
            text, voice = _text_queue.get(timeout=1.0)
        except _queue.Empty:
            continue

        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        wav = os.path.join(CACHE_DIR, f"ui_tts_{ts}.wav")
        try:
            asyncio.run(_tts_async(text, voice, wav))
        except Exception as e:
            print(f"⚠ TTS error: {e}"); continue

        if not os.path.exists(wav): continue

        phonemes    = text_to_phonemes(text)
        visemes_raw = phonemes_to_visemes(phonemes)
        visemes_m   = []
        for v in visemes_raw:
            v = 'M' if v == 'Sil' else v
            if not visemes_m or visemes_m[-1] != v: visemes_m.append(v)
        if not visemes_m: visemes_m = ['M']

        audio_dur   = len(AudioSegment.from_wav(wav)) / 1000.0
        tts_frames  = int(audio_dur * 24)   # avatar fps
        silence_per = detect_silence_periods(wav)
        fpv         = max(3, min(12, tts_frames // max(len(visemes_m), 1)))
        blink_ratio = make_blink_schedule(tts_frames, 24, silence_per)

        _audio_ready.put({
            'wav':         wav,
            'visemes':     visemes_m,
            'audio_dur':   audio_dur,
            'tts_frames':  tts_frames,
            'silence_per': silence_per,
            'fpv':         fpv,
            'blink_ratio': blink_ratio,
        })
        print(f"🎭 Ready: {len(visemes_m)} visemes, {audio_dur:.1f}s")


def _avatar_thread(video_path, voice):
    """
    Runs forever in a background thread.
    • Idle:  loops raw video frames (TTS worker handles generation in parallel)
    • Speak: blends visemes onto frames as soon as audio payload is ready
    """
    global _current_frame, _avatar_active

    face_cache = load_face_cache()
    viseme_lib = load_viseme_library()
    if not face_cache or not viseme_lib:
        print("❌ avatar thread: no cache — run Setup first"); return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ avatar thread: cannot open {video_path}"); return

    fps       = cap.get(cv2.CAP_PROP_FPS) or 24
    total_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    mouth_lib = load_mouth_library(face_cache, vid_w, vid_h)
    ox1,oy1,ox2,oy2 = face_bbox_from_cache(face_cache, vid_w, vid_h)
    mx1,my1,mx2,my2 = mouth_bbox_from_face(ox1,oy1,ox2,oy2, vid_w, vid_h, face_cache)
    ex1,ey1,ex2,ey2 = eye_bbox_from_face(ox1,oy1,ox2,oy2, vid_w, vid_h)

    frame_dur = 1.0 / fps
    _avatar_active = True

    while _avatar_active:
        # ── check for ready audio payload (non-blocking) ──────────────────────
        try:
            payload = _audio_ready.get_nowait()
        except _queue.Empty:
            payload = None

        if payload:
            # Flush any extra queued payloads (prevent repeat buildup)
            while not _audio_ready.empty():
                try: _audio_ready.get_nowait()
                except _queue.Empty: break
            # ═══ SPEAK ══════════════════════════════════════════════════
            wav         = payload['wav']
            visemes_m   = payload['visemes']
            audio_dur   = payload['audio_dur']
            tts_frames  = payload['tts_frames']
            silence_per = payload['silence_per']
            fpv         = payload['fpv']
            blink_ratio = payload['blink_ratio']

            play_audio_background(wav)
            t_start = time.perf_counter()
            last_v = prev_v = None
            smooth = 0

            for tts_fi in range(tts_frames + int(0.4 * fps)):
                if not _avatar_active: break
                ok, frame = _read_next_frame(cap, total_vid)
                if not ok: break

                t_now  = time.perf_counter() - t_start
                is_sil = any(s <= t_now <= e for s, e in silence_per)
                cur_v  = 'M' if (is_sil or t_now > audio_dur) else \
                         visemes_m[min(int(t_now * fps) // fpv, len(visemes_m)-1)]

                if cur_v != last_v:
                    prev_v, smooth, last_v = last_v, VISEME_SMOOTH_FRAMES, cur_v

                # ── mouth blend (crisp: low smooth frames) ────────────────────
                cm = mouth_lib.get(cur_v)
                if smooth > 0 and prev_v and cm is not None:
                    pm = mouth_lib.get(prev_v)
                    if pm is None: pm = cm
                    if pm.shape != cm.shape:
                        pm = cv2.resize(pm, (cm.shape[1], cm.shape[0]))
                    a  = 1.0 - smooth / max(VISEME_SMOOTH_FRAMES, 1)
                    cm = (pm.astype(np.float32) * (1-a) +
                          cm.astype(np.float32) * a).astype(np.uint8)
                    smooth = max(0, smooth - 1)

                if cm is not None:
                    frame = blend_patch(frame, cm, mx1,my1,mx2,my2, strength=0.85)

                # Blink removed — base video has natural blinks already

                cv2.putText(frame, f"{cur_v}  {t_now:.1f}s",
                            (10,24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,220,80), 1)
                _current_frame = frame

                sleep = (tts_fi+1)/fps - (time.perf_counter()-t_start)
                if sleep > 0: time.sleep(sleep)

        else:
            # ═══ IDLE ══════════════════════════════════════════════════
            t0 = time.perf_counter()
            ok, frame = _read_next_frame(cap, total_vid)
            if ok:
                cv2.putText(frame, "Listening...", (10,24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120,120,120), 1)
                _current_frame = frame
            sleep = frame_dur - (time.perf_counter() - t0)
            if sleep > 0: time.sleep(sleep)

    cap.release()


# ═══════════════════════════════════════════════════════════════════════════════
#  GRADIO UI
# ═══════════════════════════════════════════════════════════════════════════════

def build_ui():
    import gradio as gr
    DEFAULT_VIDEO = os.path.join(BASE_DIR, "Video04.mp4")

    # Start TTS worker thread (handles audio generation, never blocks avatar)
    tw = threading.Thread(target=_tts_worker, daemon=True)
    tw.start()

    # Start avatar video-loop thread
    t = threading.Thread(target=_avatar_thread,
                         args=(DEFAULT_VIDEO, "en-US-JennyNeural"),
                         daemon=True)
    t.start()
    print("🧯 TTS worker + Avatar thread started")

    def stream_avatar():
        """Generator: yields BGR→RGB frames for gr.Image streaming."""
        while True:
            f = _current_frame
            if f is not None:
                yield cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            time.sleep(1/30)   # stream at up to 30 fps

    def do_speak(text, voice):
        if not text.strip():
            return "Please enter some text."
        _text_queue.put((text.strip(), voice))   # send to TTS worker
        return f"⏳ Generating speech for: ‘{text.strip()[:50]}’…"

    def do_setup(v, f, s):
        ok = setup(v.strip(), int(f), int(s))
        return "✅ Setup complete!" if ok else "❌ Setup failed — check console"

    with gr.Blocks(title="Live Avatar Lip-Sync") as demo:
        gr.Markdown("# 🎬 Live Avatar\nThe avatar idles continuously. Type text and click Speak to lip-sync.")

        with gr.Row():
            # Left: live avatar stream
            avatar_img = gr.Image(
                label="Avatar",
                streaming=True,
                show_label=False,
                height=480,
            )
            # Right: controls
            with gr.Column():
                with gr.Accordion("⚙️ Setup (first time only)", open=False):
                    vid_in  = gr.Textbox(label="Video path", value=DEFAULT_VIDEO)
                    fidx_in = gr.Number(label="Frame index (-1=middle)", value=-1, precision=0)
                    sz_in   = gr.Radio([256,512], label="Face size", value=256)
                    s_btn   = gr.Button("⚙ Run Setup")
                    s_out   = gr.Textbox(label="Status", lines=2, interactive=False)
                    s_btn.click(do_setup, [vid_in,fidx_in,sz_in], s_out)

                txt_in   = gr.Textbox(label="💬 What should the avatar say?",
                                      lines=3, placeholder="Type here…")
                voice_in = gr.Dropdown(
                    ["en-US-JennyNeural","en-US-GuyNeural","en-GB-SoniaNeural"],
                    value="en-US-JennyNeural", label="Voice")
                speak_btn = gr.Button("🔊 Speak", variant="primary", size="lg")
                speak_out = gr.Textbox(label="Status", interactive=False)

                speak_btn.click(do_speak, [txt_in, voice_in], speak_out)

        # Start streaming as soon as page loads
        demo.load(stream_avatar, outputs=avatar_img)

    return demo


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="Real-time video lip-sync")
    ap.add_argument('--video',  default='Video04.mp4')
    ap.add_argument('--text',   default='Hello, this is a real-time lip-sync test!')
    ap.add_argument('--voice',  default='en-US-JennyNeural')
    ap.add_argument('--save',   default='', help='Also save result to this .mp4 path')
    ap.add_argument('--frame',  type=int, default=-1, help='Reference frame for setup')
    ap.add_argument('--size',   type=int, default=256)
    ap.add_argument('--setup',  action='store_true', help='Run one-time setup')
    ap.add_argument('--ui',     action='store_true', help='Launch Gradio UI')
    args = ap.parse_args()

    if args.ui:
        demo = build_ui()
        demo.launch(share=False)
    elif args.setup:
        ok = setup(args.video, args.frame, args.size)
        if ok:
            print("\n✅ Setup done. Now run:")
            print(f'  python video.py --text "Your text here."')
    else:
        if not load_face_cache():
            print("⚠  No setup found — running setup first…")
            setup(args.video, args.frame, args.size)
        realtime_lipsync(args.video, args.text, args.voice,
                         save_path=args.save)
