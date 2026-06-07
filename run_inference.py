"""Launch PPE Safety Monitor in real time using webcam input."""

import sys
from pathlib import Path

_pkg_root = Path(__file__).resolve().parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from safeguardai.detect import PPEDetector
from safeguardai.config import *


if __name__ == "__main__":
    print(f"Starting PPE Detector on device: {DEVICE}")
    print(f"Model: {MODEL_PATH}")
    print(f"Camera: {CAMERA_INDEX}")
    print("Press 'q' to exit")
    print("-" * 50)

    detector = PPEDetector(
        model_path=MODEL_PATH,
        face_cascade_path=FACE_CASCADE_PATH,
        conf_threshold=CONFIDENCE_THRESHOLD,
        device=DEVICE,
    )
    detector.run_webcam(camera_index=CAMERA_INDEX)
