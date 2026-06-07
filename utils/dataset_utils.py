import random
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CLASS_NAMES = ["with_mask", "without_mask"]


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def extract_classification_dataset(zip_path, extract_to):
    """Extract a classification dataset zip into a raw folder and print image counts."""
    zip_path = Path(zip_path)
    extract_to = Path(extract_to)
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    if extract_to.exists():
        shutil.rmtree(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as archive:
        names = archive.namelist()
        archive.extractall(extract_to)

    print(f"Extracted {len(names)} files to {extract_to}")

    data_dir = extract_to / "data"
    counts = {}
    for class_name in CLASS_NAMES:
        folder = data_dir / class_name
        counts[class_name] = sum(1 for p in folder.rglob("*") if _is_image_file(p)) if folder.exists() else 0
        print(f"  {class_name}: {counts[class_name]} images")

    return extract_to, counts


def split_classification_dataset(raw_dir, output_dir, train=0.8, val=0.1, test=0.1, seed=42):
    """Shuffle each class separately and split images into train/val/test folders."""
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    data_dir = raw_dir / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    random_gen = random.Random(seed)
    split_counts = {"train": {"with_mask": 0, "without_mask": 0}, "val": {"with_mask": 0, "without_mask": 0}, "test": {"with_mask": 0, "without_mask": 0}}

    for class_name in CLASS_NAMES:
        source_dir = data_dir / class_name
        if not source_dir.exists():
            print(f"Warning: source class folder not found: {source_dir}")
            continue

        image_files = [p for p in source_dir.iterdir() if _is_image_file(p)]
        random_gen.shuffle(image_files)
        total = len(image_files)
        train_end = int(total * train)
        val_end = train_end + int(total * val)
        splits = {
            "train": image_files[:train_end],
            "val": image_files[train_end:val_end],
            "test": image_files[val_end:],
        }

        for split_name, files in splits.items():
            dest_dir = output_dir / split_name / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for src_path in files:
                if not _is_image_file(src_path):
                    continue
                image = cv2.imread(str(src_path))
                if image is None:
                    print(f"Corrupt image skipped: {src_path}")
                    continue
                dest_path = dest_dir / src_path.name
                shutil.copy2(src_path, dest_path)
                split_counts[split_name][class_name] += 1

    print("Split      with_mask    without_mask    Total")
    for split_name in ["train", "val", "test"]:
        with_count = split_counts[split_name]["with_mask"]
        without_count = split_counts[split_name]["without_mask"]
        total = with_count + without_count
        print(f"{split_name:<10}{with_count:<13}{without_count:<15}{total}")

    return split_counts


def verify_split(dataset_dir):
    """Verify that each class subfolder in each split contains at least one image."""
    dataset_dir = Path(dataset_dir)
    ok = True
    for split_name in ["train", "val", "test"]:
        for class_name in CLASS_NAMES:
            folder = dataset_dir / split_name / class_name
            count = sum(1 for p in folder.iterdir() if _is_image_file(p)) if folder.exists() else 0
            status = "PASS" if count > 0 else "FAIL"
            print(f"{split_name}/{class_name} : {count} images - {status}")
            if count == 0:
                ok = False
    return ok


def save_sample_grid(dataset_dir, output_path, n=9):
    """Save a 3x3 grid of train images with class names overlaid."""
    dataset_dir = Path(dataset_dir)
    train_dir = dataset_dir / "train"
    image_candidates = []
    for class_name in CLASS_NAMES:
        class_dir = train_dir / class_name
        image_candidates.extend([(p, class_name) for p in class_dir.iterdir() if _is_image_file(p)])

    if not image_candidates:
        raise FileNotFoundError(f"No train images found in {train_dir}")

    sample = random.sample(image_candidates, min(n, len(image_candidates)))
    cells = []
    cell_size = 320
    for image_path, class_name in sample:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Corrupt image skipped in sample grid: {image_path}")
            continue
        text = class_name
        cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        image = cv2.resize(image, (cell_size, cell_size))
        cells.append(image)

    while len(cells) < 9:
        cells.append(np.zeros((cell_size, cell_size, 3), dtype=np.uint8))

    rows = [np.concatenate(cells[i * 3 : (i + 1) * 3], axis=1) for i in range(3)]
    grid = np.concatenate(rows, axis=0)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), grid)
    print(f"Saved sample grid to {output_path}")
    return output_path
