"""
============================================================
AI Surveillance System
Main Application - Real-Time Person Detection
============================================================
"""
import sys
import cv2
import config
from core.logger import Logger
from core.video_manager import VideoManager
from core.detector import Detector


class AISurveillanceSystem:
    def __init__(self):
        self.logger = Logger()
        self.video = VideoManager()
        self.detector = Detector()

    # ---------------------------------------------------------
    def banner(self):
        print()
        print("=" * 60)
        print(config.PROJECT_NAME)
        print("Real-Time Person Detection")
        print("=" * 60)
        print()

    # ---------------------------------------------------------
    def start(self):
        self.banner()
        self.logger.info("Application Started")

        if not self.video.open_video():
            self.logger.error("Cannot Open Video")
            return

        width = self.video.get_width()
        height = self.video.get_height()
        fps = self.video.get_fps()

        self.video.create_output(width, height, fps)

        while True:
            ret, frame = self.video.read()
            if not ret:
                self.logger.info("Video Finished")
                break

            # ----------------------------------------
            # Detection & Drawing
            # ----------------------------------------
            detections = self.detector.detect(frame)
            frame = self.detector.draw(frame, detections)

            # ----------------------------------------
            # Save Output & Display
            # ----------------------------------------
            self.video.write(frame)
            cv2.imshow(config.PROJECT_NAME, frame)

            key = cv2.waitKey(1)
            if key & 0xFF == ord("q"):
                self.logger.info("Exit Requested")
                break

        self.video.release()
        self.logger.info("Application Closed")


def main():
    app = AISurveillanceSystem()
    app.start()


if __name__ == "__main__":
    main()
