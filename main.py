"""Here’s a complete working script using the legacy solutions API in MediaPipe. This version will detect and draw face, hands, and full-body pose landmarks directly on your webcam feed:
"""
import cv2
import sys
import mediapipe as mp

# --- Setup Holistic Landmarker ---
base_options = mp.tasks.BaseOptions(model_asset_path='holistic_landmarker.task')
options = mp.tasks.vision.HolisticLandmarkerOptions(base_options=base_options)
landmarker = mp.tasks.vision.HolisticLandmarker.create_from_options(options)

# --- Camera setup ---
cap = None
for i in range(10):
    source = cv2.VideoCapture(i)
    if source.isOpened():
        print(f"Using camera index {i}")
        cap = source
        break
else:
    print("No camera found")
    sys.exit(1)

win_name = "Holistic Detection"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win_name, 640, 480)

# --- Helper function to draw landmarks ---
def draw_landmarks(frame, landmarks, color=(0,255,0)):
    if landmarks:
        h, w, _ = frame.shape
        for lm in landmarks[0]:  # take first detected set
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 2, color, -1)

# --- Main loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("Unable to capture video")
        break

    # Convert to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Wrap into MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # Run holistic detection
    result = landmarker.detect(mp_image)

    # Draw landmarks manually
    draw_landmarks(frame, result.face_landmarks, (255,0,0))     # Blue for face
    draw_landmarks(frame, result.left_hand_landmarks, (0,255,0)) # Green for left hand
    draw_landmarks(frame, result.right_hand_landmarks, (0,0,255))# Red for right hand
    draw_landmarks(frame, result.pose_landmarks, (255,255,0))    # Yellow for pose

    cv2.imshow(win_name, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()