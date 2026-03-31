import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision

import os

# Using the mediapipe tasks API
print("OpenCV version:", cv2.__version__)
print("MediaPipe version:", mp.__version__)

# Simple camera capture with OpenCV and MediaPipe
# Initialize MediaPipe Holistic
BaseOptions = mp.tasks.BaseOptions
HolisticLandmarker = vision.HolisticLandmarker
HolisticLandmarkerOptions = vision.HolisticLandmarkerOptions
VisionRunningMode = vision.RunningMode

MODEL_FILE = os.path.join(os.getcwd(), 'holistic_landmarker.task')
options = HolisticLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_FILE),
    running_mode=VisionRunningMode.IMAGE)

with HolisticLandmarker.create_from_options(options) as holistic:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Cannot read from camera")
                break

            try:
                cv2.imshow('Camera Feed', frame)
            except cv2.error:
                print("Window was closed")
                break

            # Check for 'q' key or ESC key to quit
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q') or key == 27:  # 27 is ESC key
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Camera released and window closed!")