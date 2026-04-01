import cv2
import sys
import mediapipe as mp

# Try different camera indices
# initialize holistic landmarker
base_options = mp.tasks.BaseOptions(model_asset_path='holistic_landmarker.task')
options = mp.tasks.vision.HolisticLandmarkerOptions(base_options=base_options)
with mp.tasks.vision.HolisticLandmarker.create_from_options(options) as landmarker:
    for i in range(10):
        source = cv2.VideoCapture(i)
        if source.isOpened():
            print(f"Using camera index {i}")
            break
    else:
        print("No camera found")
        sys.exit(1)

    win_name = "camera"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 640, 480)
    while True:
        ret, frame = source.read()
        if not ret:
            print("Unable to capture video")
            sys.exit(0)
        
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MP Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # Process the frame with Holistic Landmarker
        result = landmarker.detect(mp_image)
        
        # Draw face landmarks (tesselation for mesh)
        if result.face_landmarks:
            mp.tasks.vision.drawing_utils.draw_landmarks(
                frame, 
                result.face_landmarks,
                mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION
            )
        if result.left_hand_landmarks:
            mp.tasks.vision.drawing_utils.draw_landmarks(
                frame, 
                result.left_hand_landmarks,
                mp.tasks.vision.HandLandmarksConnections.HAND_LANDMARKS_TESSELATION
            )
        if result.right_hand_landmarks:
            mp.tasks.vision.drawing_utils.draw_landmarks(
                frame, 
                result.right_hand_landmarks,
                mp.tasks.vision.HandLandmarksConnections.HAND_LANDMARKS_TESSELATION
            )
        cv2.imshow(win_name, frame)
        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    source.release()
    cv2.destroyAllWindows()