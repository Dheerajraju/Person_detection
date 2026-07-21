"""
============================================================
AI Surveillance System
Configuration File (Person Detection Only)
============================================================
"""
from pathlib import Path

# ============================================================
# Project Information
# ============================================================
PROJECT_NAME = "AI Surveillance System - Person Detection"
VERSION = "1.0.0"

# ============================================================
# Directory Structure
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
VIDEOS_DIR = BASE_DIR / "videos"
INPUT_VIDEO_DIR = VIDEOS_DIR / "input"
OUTPUT_VIDEO_DIR = VIDEOS_DIR / "output"
OUTPUT_DIR = BASE_DIR / "output"
REPORT_DIR = BASE_DIR / "reports"
LOG_DIR = BASE_DIR / "logs"

# ============================================================
# Model Configuration
# ============================================================
DEFAULT_MODEL = "yolo11s.pt"
MODEL_PATH = MODELS_DIR / DEFAULT_MODEL

# ============================================================
# Video Configuration
# ============================================================
VIDEO_SOURCE = INPUT_VIDEO_DIR / "Camera_D3.mp4"

# ============================================================
# Detection Configuration
# ============================================================
CONFIDENCE_THRESHOLD = 0.50
IOU_THRESHOLD = 0.45
IMAGE_SIZE = 640
DEVICE = "cpu"

# Detect ONLY people
TARGET_CLASSES = ["person","car"]

CLASS_COLORS = {
    "person": (0, 255, 0)  # Green bounding box
}

# ============================================================
# Logging Configuration
# ============================================================
LOG_FILE = LOG_DIR / "system.log"

# ============================================================
# Create Required Folders
# ============================================================
for folder in [MODELS_DIR, INPUT_VIDEO_DIR, OUTPUT_VIDEO_DIR, OUTPUT_DIR, REPORT_DIR, LOG_DIR]:
    folder.mkdir(parents=True, exist_ok=True)
