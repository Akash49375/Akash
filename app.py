import cv2
import mediapipe as mp
import pyautogui
import math

# Disable PyAutoGUI fail-safe to prevent corner-triggered crashes
pyautogui.FAILSAFE = False
# Set a small pause between PyAutoGUI commands for stability
pyautogui.PAUSE = 0.01

# Get screen dimensions
screen_width, screen_height = pyautogui.size()

# Initialize MediaPipe Hand solution
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Configure hand tracking
# - max_num_hands: Track one hand for simplicity
# - min_detection_confidence: Minimum confidence for initial detection
# - min_tracking_confidence: Minimum confidence for tracking between frames
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Initialize webcam with multiple camera detection
cap = None
for camera_index in range(5):  # Try cameras 0-4
    test_cap = cv2.VideoCapture(camera_index)
    if test_cap.isOpened():
        ret, frame = test_cap.read()
        if ret:
            cap = test_cap
            print(f"✓ Camera {camera_index} detected and working!")
            break
        test_cap.release()

if cap is None:
    print("❌ ERROR: No working camera found!")
    print("\nTroubleshooting steps:")
    print("1. Check if camera is connected")
    print("2. Close other apps using the camera (Zoom, Skype, etc.)")
    print("3. Check camera permissions in Windows Settings")
    print("4. Try running as Administrator")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# Smoothing variables to reduce cursor jitter
smooth_factor = 5  # Higher = smoother but slower response
prev_x, prev_y = 0, 0

# Click detection variables
click_threshold = 40  # Pixel distance threshold for click detection
click_cooldown = 0  # Prevents multiple rapid clicks

print("Hand Gesture Mouse Control Started!")
print("- Move your INDEX FINGER to control the cursor")
print("- Touch THUMB and INDEX FINGER together to click")
print("- Press 'q' to quit")

while True:
    # Read frame from webcam
    success, frame = cap.read()
    if not success:
        print("Failed to capture frame")
        break
    
    # Flip frame horizontally for mirror effect (more intuitive)
    frame = cv2.flip(frame, 1)
    
    # Convert BGR (OpenCV format) to RGB (MediaPipe format)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process frame to detect hands
    results = hands.process(rgb_frame)
    
    # Get frame dimensions for coordinate conversion
    frame_height, frame_width, _ = frame.shape
    
    # Check if any hands are detected
    if results.multi_hand_landmarks:
        # Get the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Draw hand landmarks on frame for visualization
        mp_drawing.draw_landmarks(
            frame, 
            hand_landmarks, 
            mp_hands.HAND_CONNECTIONS
        )
        
        # Extract landmark positions
        # Landmark indices: 4=thumb tip, 8=index tip, 12=middle tip
        landmarks = hand_landmarks.landmark
        
        # Get index finger tip position (landmark 8)
        index_x = int(landmarks[8].x * frame_width)
        index_y = int(landmarks[8].y * frame_height)
        
        # Get thumb tip position (landmark 4)
        thumb_x = int(landmarks[4].x * frame_width)
        thumb_y = int(landmarks[4].y * frame_height)
        
        # Convert frame coordinates to screen coordinates
        # Invert Y axis for natural movement
        screen_x = int(landmarks[8].x * screen_width)
        screen_y = int(landmarks[8].y * screen_height)
        
        # Apply smoothing to reduce jitter
        smooth_x = prev_x + (screen_x - prev_x) / smooth_factor
        smooth_y = prev_y + (screen_y - prev_y) / smooth_factor
        
        # Move mouse cursor with error handling
        try:
            # Ensure coordinates are within screen bounds
            safe_x = max(0, min(smooth_x, screen_width - 1))
            safe_y = max(0, min(smooth_y, screen_height - 1))
            pyautogui.moveTo(safe_x, safe_y, duration=0)
        except Exception as e:
            pass  # Skip this frame if mouse movement fails
        
        # Update previous position for next frame
        prev_x, prev_y = smooth_x, smooth_y
        
        # Calculate distance between thumb and index finger
        distance = math.sqrt((thumb_x - index_x)**2 + (thumb_y - index_y)**2)
        
        # Draw line between thumb and index finger
        cv2.line(frame, (thumb_x, thumb_y), (index_x, index_y), (0, 255, 0), 2)
        
        # Display distance on frame
        cv2.putText(
            frame, 
            f'Distance: {int(distance)}', 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 0, 0), 
            2
        )
        
        # Detect click gesture (fingers touching)
        if distance < click_threshold and click_cooldown == 0:
            pyautogui.click()
            cv2.putText(
                frame, 
                'CLICK!', 
                (frame_width//2 - 50, frame_height//2), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.5, 
                (0, 0, 255), 
                3
            )
            click_cooldown = 15  # Cooldown frames to prevent multiple clicks
        
        # Draw cursor position indicator
        cv2.circle(frame, (index_x, index_y), 10, (255, 0, 255), -1)
        cv2.circle(frame, (thumb_x, thumb_y), 10, (255, 255, 0), -1)
    
    # Decrease cooldown counter
    if click_cooldown > 0:
        click_cooldown -= 1
    
    # Display FPS
    cv2.putText(
        frame, 
        'Press Q to quit', 
        (10, frame_height - 10), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.6, 
        (255, 255, 255), 
        2
    )
    
    # Show the frame
    cv2.imshow('Hand Gesture Mouse Control', frame)
    
    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
hands.close()