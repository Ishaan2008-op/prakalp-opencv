"""
Rock Paper Scissors Game with Holistic Face and Hand Detection
Uses MediaPipe Holistic Landmarker for face mesh and hand gesture recognition
"""
import cv2
import sys
import mediapipe as mp
import numpy as np

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

win_name = "Rock Paper Scissors - Face & Hand Detection"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win_name, 960, 720)

# --- Helper function to draw face mesh ---
def draw_face_mesh(frame, face_landmarks):
    """Draw face landmarks with tessellation mesh"""
    if face_landmarks:
        h, w, _ = frame.shape
        # Draw face mesh connections
        mp.tasks.vision.drawing_utils.draw_landmarks(
            frame,
            face_landmarks,
            mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
            mp.drawing_styles.get_default_face_landmarks_style()
        )

# --- Rock Paper Scissors Recognition Logic ---
def recognize_gesture(hand_landmarks):
    """
    Recognizes Rock, Paper, or Scissors from hand landmarks
    Returns: ('ROCK'/'PAPER'/'SCISSORS'/'NEUTRAL', confidence)
    """
    if not hand_landmarks or len(hand_landmarks) == 0:
        return 'NEUTRAL', 0.0
    
    landmarks = hand_landmarks[0]
    h, w = 480, 640  # Normalized coordinates
    
    # Extract key finger positions (tip positions)
    thumb_tip = landmarks[4]      # Thumb tip
    index_tip = landmarks[8]      # Index finger tip
    middle_tip = landmarks[12]    # Middle finger tip
    ring_tip = landmarks[16]      # Ring finger tip
    pinky_tip = landmarks[20]     # Pinky tip
    
    # Extract palm position (wrist)
    wrist = landmarks[0]
    
    # Calculate if fingers are extended (distance from wrist)
    def is_extended(tip, wrist, threshold=0.05):
        """Check if finger is extended compared to wrist position"""
        distance = np.sqrt((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2)
        return distance > threshold
    
    # Count extended fingers
    thumb_extended = is_extended(thumb_tip, wrist, 0.04)
    index_extended = is_extended(index_tip, wrist)
    middle_extended = is_extended(middle_tip, wrist)
    ring_extended = is_extended(ring_tip, wrist)
    pinky_extended = is_extended(pinky_tip, wrist)
    
    extended_count = sum([index_extended, middle_extended, ring_extended, pinky_extended])
    
    # --- Rock Paper Scissors Logic ---
    # ROCK: No fingers extended (closed fist)
    if extended_count == 0:
        return 'ROCK', 0.95
    
    # PAPER: All 4 fingers extended + thumb
    elif extended_count == 4 and thumb_extended:
        return 'PAPER', 0.90
    
    # SCISSORS: Only index and middle fingers extended
    elif extended_count == 2 and index_extended and middle_extended and not ring_extended and not pinky_extended:
        # Check if they form a V shape (not touching)
        distance = np.sqrt((index_tip.x - middle_tip.x)**2 + (index_tip.y - middle_tip.y)**2)
        if distance > 0.02:  # Fingers should be apart
            return 'SCISSORS', 0.90
    
    return 'NEUTRAL', 0.5

# --- Main game loop ---
frame_count = 0
gesture_history = []
confidence_threshold = 0.7

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Unable to capture video")
            break
        
        frame_count += 1
        
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # Run holistic detection
        result = landmarker.detect(mp_image)
        
        # --- Draw Face Mesh ---
        if result.face_landmarks:
            draw_face_mesh(frame, result.face_landmarks)
            cv2.putText(frame, "Face Detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # --- Process Hand Gestures ---
        left_gesture = 'NEUTRAL'
        right_gesture = 'NEUTRAL'
        left_confidence = 0.0
        right_confidence = 0.0
        
        # Left hand detection
        if result.left_hand_landmarks:
            left_gesture, left_confidence = recognize_gesture(result.left_hand_landmarks)
            h, w, _ = frame.shape
            
            # Draw left hand skeleton
            mp.tasks.vision.drawing_utils.draw_landmarks(
                frame,
                result.left_hand_landmarks,
                mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS,
                mp.drawing_styles.get_default_hand_landmarks_style()
            )
            
            # Display left hand gesture
            if left_confidence >= confidence_threshold:
                cv2.putText(frame, f"LEFT: {left_gesture} ({left_confidence:.2f})", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Right hand detection
        if result.right_hand_landmarks:
            right_gesture, right_confidence = recognize_gesture(result.right_hand_landmarks)
            h, w, _ = frame.shape
            
            # Draw right hand skeleton
            mp.tasks.vision.drawing_utils.draw_landmarks(
                frame,
                result.right_hand_landmarks,
                mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS,
                mp.drawing_styles.get_default_hand_landmarks_style()
            )
            
            # Display right hand gesture
            if right_confidence >= confidence_threshold:
                cv2.putText(frame, f"RIGHT: {right_gesture} ({right_confidence:.2f})", 
                           (w - 350, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # --- Game Instructions ---
        cv2.putText(frame, "Rock Paper Scissors Recognition", 
                   (w // 2 - 200, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, "Press 'q' to quit", 
                   (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        # Display the frame
        cv2.imshow(win_name, frame)
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nInterrupted by user")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Game ended")
