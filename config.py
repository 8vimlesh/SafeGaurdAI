"""Configuration loader for PPE Safety System.

Reads environment variables from a .env file and exposes configuration
constants for the application. Uses pathlib.Path for all filesystem paths.
"""

from pathlib import Path
import os

import cv2
import torch
from dotenv import load_dotenv

# Load .env from project root next to this file
HERE = Path(__file__).resolve().parent
DOTENV_PATH = HERE / ".env"
load_dotenv(dotenv_path=DOTENV_PATH)

# Application paths and settings with sensible defaults
MODEL_PATH = "runs/classify/runs/train/face_mask_classifier-2/weights/best.pt"
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "224"))
SCREENSHOT_DIR = Path(os.getenv("SCREENSHOT_DIR") or (HERE / "static" / "screenshots"))
FACE_CASCADE_PATH = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///ppe_safety.db")

# Ensure directories exist
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

__all__ = [
    "MODEL_PATH",
    "CAMERA_INDEX",
    "CONFIDENCE_THRESHOLD",
    "IMAGE_SIZE",
    "SCREENSHOT_DIR",
    "FACE_CASCADE_PATH",
    "DEVICE",
    "DATABASE_URL",
]
