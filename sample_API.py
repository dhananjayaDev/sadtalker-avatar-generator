"""
Sample REST API for SadTalker Live Avatar (AWS-hostable).

Dependency: pip install flask

Run locally:   python sample_API.py
Run on AWS:    gunicorn -w 1 -b 0.0.0.0:8000 sample_API:app
               (use 1 worker due to GPU/TTS; scale via more instances)

Env vars:
  API_KEY       Optional. If set, require X-API-Key header. If empty, no auth.
  PORT          Default 8000.
"""

import os
import sys

# Run from directory containing live.py so BASE_DIR and cache paths resolve
APP_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(APP_DIR)
sys.path.insert(0, APP_DIR)

from flask import Flask, request, jsonify, send_file
from live import compose_live_video, RESULT_DIR

app = Flask(__name__)

REQUIRED_API_KEY = os.environ.get("API_KEY", "").strip()
PORT = int(os.environ.get("PORT", "8000"))


def require_api_key():
    if not REQUIRED_API_KEY:
        return None
    key = request.headers.get("X-API-Key") or request.headers.get("Authorization", "").replace("Bearer ", "")
    if key != REQUIRED_API_KEY:
        return jsonify({"error": "Invalid or missing API key"}), 401
    return None


@app.route("/health", methods=["GET"])
def health():
    """For AWS ALB/ECS health checks."""
    return jsonify({"status": "ok"}), 200


@app.route("/v1/generate", methods=["POST"])
def generate():
    """
    Generate avatar video from text.
    Body (JSON): { "text": "Hello world", "fps": 25 }
    Returns: video file (attachment) or JSON error.
    """
    err = require_api_key()
    if err is not None:
        return err

    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Missing or empty 'text'"}), 400

    fps = int(data.get("fps", 25))
    fps = max(10, min(60, fps))

    try:
        video_path, audio_path, status = compose_live_video(text, fps=fps)
    except Exception as e:
        return jsonify({"error": "Generation failed", "detail": str(e)}), 500

    if not video_path or not os.path.exists(video_path):
        return jsonify({"error": "No video produced", "message": status}), 500

    # Return video as download; on AWS you would typically upload to S3 and return a URL
    return send_file(
        video_path,
        mimetype="video/mp4",
        as_attachment=True,
        download_name=os.path.basename(video_path),
    )


@app.route("/v1/generate/status", methods=["POST"])
def generate_status():
    """
    Same as /v1/generate but returns JSON with paths (for server-side use).
    Body: { "text": "...", "fps": 25 }
    Response: { "video_path": "...", "audio_path": "...", "message": "..." }
    """
    err = require_api_key()
    if err is not None:
        return err

    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Missing or empty 'text'"}), 400

    fps = int(data.get("fps", 25))
    fps = max(10, min(60, fps))

    try:
        video_path, audio_path, status = compose_live_video(text, fps=fps)
    except Exception as e:
        return jsonify({"error": "Generation failed", "detail": str(e)}), 500

    if not video_path or not os.path.exists(video_path):
        return jsonify({"error": "No video produced", "message": status}), 500

    return jsonify({
        "video_path": video_path,
        "audio_path": audio_path or "",
        "message": status,
    }), 200


if __name__ == "__main__":
    print(f"Starting API on port {PORT} (API_KEY={'set' if REQUIRED_API_KEY else 'not set'})")
    app.run(host="0.0.0.0", port=PORT, debug=False)
