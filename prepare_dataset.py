import shutil
import zipfile
from datetime import datetime
from pathlib import Path

from safeguardai.utils.dataset_utils import (
    extract_classification_dataset,
    save_sample_grid,
    split_classification_dataset,
    verify_split,
)

ZIP_PATH = Path(r"D:\Dataset.zip")
PROJECT_ROOT = Path(__file__).resolve().parent
EXTRACT_ROOT = PROJECT_ROOT / "dataset_raw"
DATA_ROOT = PROJECT_ROOT / "dataset"
SAMPLE_OUTPUT = PROJECT_ROOT / "static" / "samples" / "train_samples.png"
CLASS_NAMES = ["with_mask", "without_mask"]


def _timestamped_step(step, message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [STEP {step}/5] {message}")


def _overwrite_data_yaml():
    data_yaml = PROJECT_ROOT / "data.yaml"
    content = (
        "yamlpath: ./dataset\n"
        "train: train\n"
        "val: val\n"
        "test: test\n"
        "nc: 2\n"
        "names:\n"
        "  0: with_mask\n"
        "  1: without_mask\n"
    )
    data_yaml.write_text(content, encoding="utf-8")
    return data_yaml


def _count_raw_images(raw_dir):
    data_dir = raw_dir / "data"
    counts = {"with_mask": 0, "without_mask": 0}
    for class_name in CLASS_NAMES:
        class_dir = data_dir / class_name
        if class_dir.exists():
            counts[class_name] = sum(1 for p in class_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    return counts


def main():
    _timestamped_step(1, f"Extract {ZIP_PATH} -> dataset_raw/")
    if not ZIP_PATH.exists():
        print(f"ERROR: Could not find {ZIP_PATH} — please check the path and exit")
        return

    if EXTRACT_ROOT.exists():
        shutil.rmtree(EXTRACT_ROOT)
    EXTRACT_ROOT.mkdir(parents=True, exist_ok=True)
    extract_classification_dataset(ZIP_PATH, EXTRACT_ROOT)

    _timestamped_step(2, "Count images per class")
    raw_counts = _count_raw_images(EXTRACT_ROOT)
    print(f"  with_mask images    : {raw_counts['with_mask']}")
    print(f"  without_mask images : {raw_counts['without_mask']}")

    split_needed = True
    train_with_mask = DATA_ROOT / "train" / "with_mask"
    if train_with_mask.exists() and any(train_with_mask.iterdir()):
        split_needed = False
        print("Dataset already split. Delete dataset/ folder to re-run.")

    if split_needed:
        _timestamped_step(3, "Split into train/val/test -> dataset/")
        split_counts = split_classification_dataset(EXTRACT_ROOT, DATA_ROOT, train=0.8, val=0.1, test=0.1, seed=42)
        _timestamped_step(4, "Verify all folders are populated")
        verify_split(DATA_ROOT)
    else:
        _timestamped_step(3, "Split into train/val/test -> dataset/")
        print("Skipping split because dataset already exists.")
        _timestamped_step(4, "Verify all folders are populated")
        verify_split(DATA_ROOT)
        split_counts = {
            "train": {
                "with_mask": sum(1 for p in (DATA_ROOT / "train" / "with_mask").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}),
                "without_mask": sum(1 for p in (DATA_ROOT / "train" / "without_mask").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}),
            },
            "val": {
                "with_mask": sum(1 for p in (DATA_ROOT / "val" / "with_mask").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}),
                "without_mask": sum(1 for p in (DATA_ROOT / "val" / "without_mask").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}),
            },
            "test": {
                "with_mask": sum(1 for p in (DATA_ROOT / "test" / "with_mask").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}),
                "without_mask": sum(1 for p in (DATA_ROOT / "test" / "without_mask").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}),
            },
        }

    _timestamped_step(5, "Save 3x3 sample grid -> static/samples/train_samples.png")
    save_sample_grid(DATA_ROOT, SAMPLE_OUTPUT, n=9)

    data_yaml_path = _overwrite_data_yaml()
    train_total = split_counts["train"]["with_mask"] + split_counts["train"]["without_mask"]
    val_total = split_counts["val"]["with_mask"] + split_counts["val"]["without_mask"]
    test_total = split_counts["test"]["with_mask"] + split_counts["test"]["without_mask"]
    total_processed = train_total + val_total + test_total

    print("==========================================")
    print("  DATASET PREPARATION COMPLETE")
    print("==========================================")
    print(f"  with_mask images    : {raw_counts['with_mask']}")
    print(f"  without_mask images : {raw_counts['without_mask']}")
    print(f"  Total               : {total_processed}")
    print(f"  Train               : {train_total}")
    print(f"  Val                 : {val_total}")
    print(f"  Test                : {test_total}")
    print("  Classes             : with_mask, without_mask")
    print("  data.yaml           : ready")
    print(f"  Sample grid         : {SAMPLE_OUTPUT.as_posix()}")
    print("==========================================")
    print("  DATASET READY FOR TRAINING")
    print("==========================================")


if __name__ == "__main__":
    main()
