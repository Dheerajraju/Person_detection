"""
============================================================
AI Surveillance System
Detection Engine (Person Detection Only)
============================================================
"""
import cv2
import config
from core.logger import Logger
from core.model_loader import ModelLoader


class Detector:
    def __init__(self):
        self.logger = Logger()
        self.model_loader = ModelLoader()
        self.model = self.model_loader.load()
        self.names = self.model.names
        self.logger.info("Detector Initialized for Person Detection")

    # -----------------------------------------------------
    def detect(self, frame):
        detections = []
        results = self.model.predict(
            source=frame,
            conf=config.CONFIDENCE_THRESHOLD,
            iou=config.IOU_THRESHOLD,
            imgsz=config.IMAGE_SIZE,
            device=config.DEVICE,
            verbose=False,
        )

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                name = self.names[cls]

                # Strictly detect only persons
                if name != "person":
                    continue

                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({
                    "class": name,
                    "confidence": conf,
                    "box": (x1, y1, x2, y2),
                })

        return detections

    # -----------------------------------------------------
    def draw(self, frame, detections):
        for obj in detections:
            x1, y1, x2, y2 = obj["box"]
            confidence = obj["confidence"]
            color = (0, 255, 0)  # Green

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label text
            label = f"Person {confidence:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

            # Background rectangle for label readability
            cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w + 10, y1), color, -1)
            cv2.putText(
                frame,
                label,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )

        return frame
