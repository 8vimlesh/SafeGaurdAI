"""Verification script for PPE Safety System scaffold.

When executed, this script checks Python version, required packages,
creates missing folders, attempts to detect a CUDA-capable GPU, and
validates that `data.yaml` is readable.
"""

import sys
import subprocess
import shutil
from pathlib import Path
from importlib import import_module

ROOT = Path(__file__).resolve().parent

REQUIRED_PACKAGES = [
    "ultralytics",
    "opencv-python",
    "flask",
    "flask_socketio",
    "pandas",
    "numpy",
    "matplotlib",
    "PIL",
    "python_dotenv",
    "sqlalchemy",
]

REQUIRED_DIRS = [
    ROOT / "dataset" / "images" / "train",
    ROOT / "dataset" / "images" / "val",
    ROOT / "dataset" / "images" / "test",
    ROOT / "dataset" / "labels" / "train",
    ROOT / "dataset" / "labels" / "val",
    ROOT / "dataset" / "labels" / "test",
    ROOT / "models",
    ROOT / "runs",
    ROOT / "static" / "screenshots",
    ROOT / "static" / "css",
    ROOT / "templates",
    ROOT / "utils",
    ROOT / "logs",
]


def check_python_version() -> bool:
    ok = sys.version_info >= (3, 8)
    print(f"Python version: {sys.version.split()[0]} -> {'OK' if ok else 'FAIL'}")
    return ok


def check_packages() -> dict:
    results = {}
    print("\nChecking required packages:")
    for pkg in REQUIRED_PACKAGES:
        try:
            # Try to map human package names to importable modules
            if pkg == "opencv-python":
                mod = "cv2"
            elif pkg == "PIL":
                mod = "PIL"
            elif pkg == "python_dotenv":
                mod = "dotenv"
            elif pkg == "flask_socketio":
                mod = "flask_socketio"
            else:
                mod = pkg
            m = import_module(mod)
            ver = getattr(m, "__version__", None)
            if not ver:
                # try importlib.metadata
                try:
                    from importlib.metadata import version

                    ver = version(pkg)
                except Exception:
                    ver = "unknown"
            print(f" - {pkg}: installed (module {mod}, version {ver})")
            results[pkg] = True
        except Exception:
            print(f" - {pkg}: MISSING")
            results[pkg] = False
    return results


def ensure_dirs():
    created = []
    for d in REQUIRED_DIRS:
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
            created.append(d)
    if created:
        print("\nCreated missing directories:")
        for c in created:
            print(f" - {c}")
    else:
        print("\nAll required directories already exist.")


def detect_cuda() -> str:
    # Try torch if available
    try:
        import torch

        cuda = torch.cuda.is_available()
        return f"torch detected: CUDA available = {cuda}"
    except Exception:
        pass

    # Try nvidia-smi
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(["nvidia-smi", "-L"], stderr=subprocess.DEVNULL, text=True)
            return f"nvidia-smi found: {out.strip()}"
        except Exception:
            return "nvidia-smi present but failed to query GPU"

    return "No GPU detection available (torch not installed and nvidia-smi not found)"


def validate_data_yaml() -> bool:
    yaml_path = ROOT / "data.yaml"
    if not yaml_path.exists():
        print("data.yaml: MISSING")
        return False
    # Try to import yaml if available
    try:
        import yaml

        with open(yaml_path, "r", encoding="utf-8") as fh:
            _ = yaml.safe_load(fh)
        print("data.yaml: OK (parsed successfully)")
        return True
    except Exception as e:
        print(f"data.yaml: could not validate (yaml package missing or parse error): {e}")
        return False


def main():
    failures = []
    print("Verifying PPE Safety System scaffold\n")
    if not check_python_version():
        failures.append("Python>=3.8 required")

    pkg_results = check_packages()
    if not all(pkg_results.values()):
        failures.append("Missing Python packages (see list above)")

    ensure_dirs()

    print("\nDetecting CUDA-capable GPU:")
    print(" ", detect_cuda())

    if not validate_data_yaml():
        failures.append("data.yaml missing or invalid")

    print("\nSummary:")
    if not failures:
        print("Setup complete ✓")
        return 0
    else:
        print("Setup incomplete. Failures:")
        for f in failures:
            print(" - ", f)
        return 2


if __name__ == "__main__":
    sys.exit(main())
