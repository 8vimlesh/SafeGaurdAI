"""Real-time PPE face mask detection and classification."""

from pathlib import Path
import time

import cv2
import torch
from ultralytics import YOLO

from safeguardai.utils.detection_logger import save_violation_screenshot
from safeguardai.utils.database import log_violation, log_detection_log
from safeguardai.utils.audio_alert import play_alarm, is_critical_violation
import safeguardai.config as config
import threading


class PPEDetector:
    """PPE detector wrapper around a YOLOv8 classifier and OpenCV face detector."""

    def __init__(self, model_path, face_cascade_path, conf_threshold=0.5, device="cpu"):
        self.model_path = Path(model_path)
        self.face_cascade_path = Path(face_cascade_path)
        self.conf_threshold = float(conf_threshold)
        self.device = str(device).lower()
        if self.device == "gpu":
            self.device = "cuda"
        if self.device == "cuda" and not torch.cuda.is_available():
            print("[WARNING] CUDA unavailable, falling back to CPU.")
            self.device = "cpu"

        self.model = YOLO(str(self.model_path))
        self.face_cascade = cv2.CascadeClassifier(str(self.face_cascade_path))
        if self.face_cascade.empty():
            raise ValueError(f"Unable to load face cascade from {self.face_cascade_path}")

        self.last_face_count = 0
        self.last_frame_time = time.time()
        self.fps = 0.0

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )
        return [tuple(map(int, face)) for face in faces] if len(faces) else []

    def classify_faces(self, frame, face_boxes):
        results = []
        for x, y, w, h in face_boxes:
            x, y, w, h = int(x), int(y), int(w), int(h)
            face_crop = frame[y : y + h, x : x + w]
            if face_crop.size == 0:
                continue

            inference = self.model(face_crop, device=self.device)
            if not inference:
                continue

            result = inference[0]
            class_id = int(result.probs.top1)
            confidence = float(result.probs.top1conf)
            class_name = result.names.get(class_id, str(class_id))
            is_violation = class_name == "without_mask" and confidence >= self.conf_threshold

            results.append(
                {
                    "bbox": (x, y, w, h),
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "is_violation": is_violation,
                }
            )

        return results

    def process_frame(self, frame):
        face_boxes = self.detect_faces(frame)
        face_results = self.classify_faces(frame, face_boxes)
        self.last_face_count = len(face_results)

        annotated = frame.copy()
        has_violation = False

        for face_info in face_results:
            x, y, w, h = face_info["bbox"]
            confidence = face_info["confidence"]
            class_name = face_info["class_name"]
            is_violation = face_info["is_violation"]

            if class_name == "with_mask" and confidence >= self.conf_threshold:
                color = (0, 255, 0)
                label = "✓ WITH MASK"
            elif class_name == "without_mask" and confidence >= self.conf_threshold:
                color = (0, 0, 255)
                label = "✗ WITHOUT MASK"
            else:
                color = (0, 255, 255)
                label = f"UNCERTAIN ({confidence:.2f})"

            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            text_origin = (x, y - 10 if y - 10 > 15 else y + 20)
            cv2.putText(
                annotated,
                label,
                text_origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

            if is_violation:
                has_violation = True

        if has_violation:
            cv2.rectangle(
                annotated,
                (0, 0),
                (annotated.shape[1] - 1, annotated.shape[0] - 1),
                (0, 0, 255),
                4,
            )

        violations = [face_info for face_info in face_results if face_info["is_violation"]]
        return annotated, violations

    def run_webcam(self, camera_index=0):
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"[ERROR] Unable to open webcam at index {camera_index}.")
            return

        frame_counter = 0

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[ERROR] Webcam disconnected or failed to capture frame.")
                break

            frame_counter += 1
            annotated_frame, violations = self.process_frame(frame)
            current_time = time.time()
            elapsed = current_time - self.last_frame_time
            self.fps = 1.0 / elapsed if elapsed > 0 else 0.0
            self.last_frame_time = current_time

            cv2.putText(
                annotated_frame,
                f"FPS: {self.fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            camera_id = f"CAM-{camera_index:02d}"
            if violations:
                screenshot_path = None
                if frame_counter % 30 == 0:
                    screenshot_path = save_violation_screenshot(annotated_frame, violations)
                for violation in violations:
                    try:
                        log_violation(
                            camera_id,
                            "Zone-A",
                            violation["class_name"],
                            violation["confidence"],
                            screenshot_path or "",
                        )
                    except Exception as e:
                        print(f"[DB ERROR] {e}")
                    # Play audible alarm for critical violations
                    try:
                        if getattr(config, 'AUDIO_ALERT_ENABLED', False) and is_critical_violation(violation["class_name"]):
                            # duration for play_alarm expects seconds; config.ALARM_DURATION is ms
                            dur_seconds = float(getattr(config, 'ALARM_DURATION', 2000)) / 1000.0
                            print(f"[ALARM] Triggering alarm for {violation['class_name']} ({dur_seconds}s)")
                            threading.Thread(target=play_alarm, args=(dur_seconds,), daemon=True).start()
                    except Exception as e:
                        print(f"[ALARM ERROR] {e}")

            if frame_counter % 30 == 0:
                try:
                    log_detection_log(camera_id, self.last_face_count, len(violations))
                except Exception as e:
                    print(f"[DB ERROR] {e}")

            print(f"[FRAME {frame_counter}] {self.last_face_count} faces detected | {len(violations)} violation")
            for violation in violations:
                print(f"  ✗ {violation['class_name'].upper().replace('_', ' ')} ({violation['confidence']:.2f} conf)")
            if frame_counter % 30 == 0:
                print(f"[FPS: {self.fps:.1f}]")

            cv2.imshow("PPE Safety Monitor", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
