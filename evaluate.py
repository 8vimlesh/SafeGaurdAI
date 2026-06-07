"""YOLOv8 Classification Model Evaluation Script.

Evaluates the trained model on validation/test data and runs inference
on sample images to verify predictions.
"""

from pathlib import Path
import sys

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed. Install with: pip install ultralytics")
    sys.exit(1)


def evaluate_classification_model():
    """Evaluate the trained YOLOv8 classification model.
    
    Runs validation metrics and inference on test images.
    """
    project_root = Path(__file__).resolve().parent
    dataset_path = project_root / "dataset"
    model_path = project_root / "runs" / "train" / "face_mask_classifier" / "weights" / "best.pt"
    test_path = dataset_path / "test"
    
    # Verify paths exist
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Please run 'python train.py' first to train the model.")
        sys.exit(1)
    
    if not test_path.exists():
        print(f"ERROR: Test dataset not found at {test_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("Starting YOLOv8 Classification Evaluation")
    print("=" * 60)
    
    # Load the best trained model
    model = YOLO(str(model_path))
    
    # Run validation
    print("\nRunning validation metrics...")
    metrics = model.val(data=str(dataset_path))
    
    # Extract accuracy metrics from results
    top1_acc = metrics.results_dict.get("top1_acc", 0.0) * 100 if metrics.results_dict else 0.0
    top5_acc = metrics.results_dict.get("top5_acc", 0.0) * 100 if metrics.results_dict else 0.0
    loss = metrics.results_dict.get("loss", 0.0) if metrics.results_dict else 0.0
    
    # Print evaluation table
    print("\n" + "=" * 60)
    print("  MODEL EVALUATION")
    print("=" * 60)
    print(f"  Top-1 Accuracy  : {top1_acc:.1f}%")
    print(f"  Top-5 Accuracy  : {top5_acc:.1f}%")
    print(f"  Loss            : {loss:.3f}")
    print("=" * 60)
    
    # Run inference on test images
    print("\nRunning inference on test images...")
    results = model.predict(source=str(test_path), conf=0.5, verbose=False)
    
    print("\nSample Predictions (first 5 images):")
    print("-" * 60)
    for i, r in enumerate(results[:5], 1):
        if r.probs is not None:
            top1_class = r.probs.top1c
            top1_conf = r.probs.top1conf
            class_names = r.names
            class_name = class_names[top1_class] if top1_class < len(class_names) else "Unknown"
            image_name = Path(r.path).name
            print(f"{i}. Image: {image_name:30s} → {class_name:15s} ({top1_conf:.2f})")
    print("-" * 60)
    
    print("\nEvaluation Complete!")


if __name__ == "__main__":
    evaluate_classification_model()
