"""Detection logging helpers for PPE Safety System.

This module wraps database functions to log violations and detection stats,
and saves violation screenshots when required.
"""

from datetime import datetime
from pathlib import Path
from typing import List

import cv2

from safeguardai.utils.database import log_violation as db_log_violation, log_detection_log as db_log_detection_log
from safeguardai.config import SCREENSHOT_DIR

SCREENSHOT_DIR = Path(SCREENSHOT_DIR)
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)


def log_violation(class_name: str, confidence: float, screenshot_path: str, camera_id: str = "CAM-01", zone: str = "Zone-A"):
    """Log a detected violation to the database and print a summary."""
    db_log_violation(camera_id, zone, class_name, confidence, screenshot_path)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[VIOLATION] {class_name} at {timestamp} (conf: {confidence:.2f})")


def log_frame_stats(camera_id: str, total_faces: int, violations_count: int):
    """Log a frame-level summary of detection statistics."""
    db_log_detection_log(camera_id, total_faces, violations_count)
    print(f"[STATS] camera={camera_id} total_faces={total_faces} violations={violations_count}")


def save_violation_screenshot(frame, violation_list: List[dict]):
    """Save a screenshot when violations are detected and return the saved path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"violation_{timestamp}.jpg"
    screenshot_path = SCREENSHOT_DIR / filename
    cv2.imwrite(str(screenshot_path), frame)
    return str(screenshot_path)
