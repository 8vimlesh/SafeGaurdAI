"""YOLOv8 Classification Model Training Script.

Trains a nano YOLOv8 classification model on the face mask detection dataset.
Handles CUDA errors gracefully with CPU fallback.
"""

from pathlib import Path
import sys

try:
    from ultralytics import YOLO
    import torch
except ImportError:
    print("ERROR: ultralytics or torch not installed. Install with: pip install ultralytics")
    sys.exit(1)


def train_classification_model():
    """Train YOLOv8 nano classification model with specified parameters.
    
    Falls back to CPU if CUDA unavailable.
    """
    project_root = Path(__file__).resolve().parent
    dataset_path = project_root / "dataset"
    
    # Verify dataset exists
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found at {dataset_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("Starting YOLOv8 Classification Training")
    print("=" * 60)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        device = 0
        print(f"CUDA available: Using GPU device {device}")
    else:
        device = "cpu"
        print("WARNING: CUDA not available. Training will use CPU.")
    
    # Load pretrained nano classification model
    model = YOLO("yolov8n-cls.pt")
    
    # Train with specified parameters
    results = model.train(
        data=str(dataset_path),      # YOLOv8 classification expects folder structure
        epochs=50,
        imgsz=224,                   # classification default
        batch=32,                    # RTX 2050 can handle this
        name="face_mask_classifier",
        project="runs/train",
        patience=10,                 # early stopping after 10 epochs no improvement
        optimizer="AdamW",
        lr0=0.001,
        augment=True,
        device=device,               # Auto-detected GPU or CPU
        val=True,
        save=True,
        plots=True,
        verbose=True
    )
    
    best_model_path = project_root / results.save_dir / "weights" / "best.pt"
    print("\n" + "=" * 60)
    print(f"Best model saved at: {best_model_path}")
    print("=" * 60)


if __name__ == "__main__":
    train_classification_model()
