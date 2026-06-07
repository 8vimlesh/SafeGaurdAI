"""Flask web dashboard for real-time PPE safety monitoring.

Provides a web interface for live video streaming, violation alerts via SocketIO,
and compliance statistics. Integrates with detect.py for frame processing.
"""

import os
import sys
from pathlib import Path

_project_dir = Path(__file__).resolve().parent
os.chdir(_project_dir)

_pkg_root = _project_dir.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

import threading
import cv2
import time
from collections import deque
from datetime import datetime

from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO, emit
import torch

from safeguardai.detect import PPEDetector
from safeguardai.config import MODEL_PATH, FACE_CASCADE_PATH, CONFIDENCE_THRESHOLD, DEVICE, CAMERA_INDEX
from safeguardai.utils.database import log_violation, log_detection_log, get_recent_violations, get_compliance_stats, init_db
from safeguardai.utils.audio_alert import play_alarm, is_critical_violation
import safeguardai.config as config

# Initialize Flask app
app = Flask(__name__, template_folder=str(_project_dir / "templates"))
app.config["SECRET_KEY"] = "ppe_safety_secret_2026"
socketio = SocketIO(app, cors_allowed_origins="*", ping_timeout=10, ping_interval=5)

# Global state
detector = None
camera = None
frame_lock = threading.Lock()
current_frame = None
violation_history = deque(maxlen=20)
running = True


def init_database():
    """Initialize database with error handling."""
    try:
        init_db()
        print("[DB] Database initialized successfully")
    except Exception as e:
        print(f"[DB WARNING] Could not initialize database: {e}")


def init_detector():
    """Initialize the PPE detector."""
    global detector
    try:
        print(f"[DETECTOR] Loading model from {MODEL_PATH}")
        detector = PPEDetector(
            model_path=str(MODEL_PATH),
            face_cascade_path=str(FACE_CASCADE_PATH),
            conf_threshold=CONFIDENCE_THRESHOLD,
            device=DEVICE,
        )
        print("[DETECTOR] Model loaded successfully")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to initialize detector: {e}")
        return False


def capture_and_process():
    """Continuous frame capture and processing loop."""
    global current_frame, camera, running

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Unable to open webcam at index {CAMERA_INDEX}")
        return

    cap.set(cv2.CAP_PROP_FPS, 20)
    frame_counter = 0
    last_db_log_time = time.time()

    while running:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[WARNING] Failed to capture frame")
            time.sleep(0.1)
            continue

        frame_counter += 1

        try:
            annotated_frame, violations = detector.process_frame(frame)

            with frame_lock:
                current_frame = annotated_frame

            if violations:
                camera_id = f"CAM-{CAMERA_INDEX:02d}"
                
                for violation in violations:
                    violation_info = {
                        "timestamp": datetime.now().isoformat(),
                        "class_name": violation["class_name"],
                        "confidence": f"{violation['confidence']:.2f}",
                        "camera_id": camera_id,
                    }
                    violation_history.append(violation_info)

                    try:
                        log_violation(
                            camera_id,
                            "Zone-A",
                            violation["class_name"],
                            violation["confidence"],
                            "",
                        )
                    except Exception as e:
                        pass

                    # Play audible alarm for critical violations (background thread)
                    try:
                        if getattr(config, 'AUDIO_ALERT_ENABLED', False) and is_critical_violation(violation["class_name"]):
                            dur_seconds = float(getattr(config, 'ALARM_DURATION', 2000)) / 1000.0
                            print(f"[ALARM] Triggering alarm for {violation['class_name']} ({dur_seconds}s)")
                            threading.Thread(target=play_alarm, args=(dur_seconds,), daemon=True).start()
                    except Exception as e:
                        print(f"[ALARM ERROR] {e}")

                socketio.emit(
                    "violation_alert",
                    {
                        "violations": list(violation_history),
                        "timestamp": datetime.now().isoformat(),
                    },
                    broadcast=True,
                    namespace="/",
                )

            if frame_counter % 30 == 0:
                try:
                    log_detection_log(camera_id, detector.last_face_count, len(violations))
                except Exception:
                    pass

        except Exception as e:
            print(f"[PROCESS ERROR] {e}")

        time.sleep(0.05)

    cap.release()


@app.route("/")
def index():
    """Render the dashboard."""
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """Stream video frames as MJPEG."""
    def generate():
        while running:
            with frame_lock:
                if current_frame is None:
                    time.sleep(0.01)
                    continue
                frame = current_frame.copy()

            ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-length: " + str(len(buffer)).encode() + b"\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )
            time.sleep(0.05)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/violations")
def get_violations():
    """Get recent violations."""
    try:
        return jsonify({"violations": list(violation_history)})
    except Exception as e:
        return jsonify({"violations": [], "error": str(e)})


@app.route("/api/stats")
def get_stats():
    """Get compliance statistics."""
    try:
        stats = get_compliance_stats()
        return jsonify(
            {
                "compliance_pct": stats.get("avg_compliance_pct", 0.0),
                "total_violations": stats.get("records", 0),
            }
        )
    except Exception:
        return jsonify({"compliance_pct": 0.0, "total_violations": 0})


@app.route("/api/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "device": DEVICE})


@socketio.on("connect")
def handle_connect():
    """Handle client connection."""
    emit("violation_alert", {"violations": list(violation_history)})
    print("[SOCKETIO] Client connected")


@socketio.on("disconnect")
def handle_disconnect():
    """Handle client disconnection."""
    print("[SOCKETIO] Client disconnected")


if __name__ == "__main__":
    print("[APP] Starting PPE Safety Monitor Dashboard")
    print(f"[APP] Model: {MODEL_PATH}")
    print(f"[APP] Device: {DEVICE}")
    print(f"[APP] Camera: {CAMERA_INDEX}")

    init_database()

    if not init_detector():
        print("[ERROR] Failed to initialize detector. Exiting.")
        exit(1)

    thread = threading.Thread(target=capture_and_process, daemon=True)
    thread.start()

    print("[APP] Starting Flask SocketIO server...")
    socketio.run(
        app,
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False,
        allow_unsafe_werkzeug=True,
    )
