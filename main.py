import cv2
import mediapipe as mp
import pyautogui
import math

# Initialize MediaPipe Hand Detection
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Get screen size
screen_w, screen_h = pyautogui.size()

# Initialize webcam
cap = cv2.VideoCapture(0)

# Smooth mouse motion factor
smooth_factor = 5
prev_x, prev_y = 0, 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame for natural interaction
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape 

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get coordinates of important points
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            # Convert normalized coordinates to pixel coordinates
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)

            # Map to screen coordinates
            screen_x = screen_w * thumb_tip.x
            screen_y = screen_h * thumb_tip.y

            # Smooth mouse movement
            curr_x = prev_x + (screen_x - prev_x) / smooth_factor # prevents sudden jumps with cursor
            curr_y = prev_y + (screen_y - prev_y) / smooth_factor
            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y

            # Calculate distance between thumb and index
            dist = math.hypot(index_x - thumb_x, index_y - thumb_y)

            # CLICK: Thumb and index close together
            if dist < 30: # it is considered a click if the thumb and index are within 30 pixels away
                cv2.putText(frame, 'Click!', (thumb_x, thumb_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                pyautogui.click()

            # SCROLL: Move thumb up or down relative to middle finger
            thumb_middle_dist = thumb_y - int(middle_tip.y * h)
            if thumb_middle_dist > 60:
                pyautogui.scroll(50)  # scroll up
                cv2.putText(frame, 'Scroll Up', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            elif thumb_middle_dist < -60:
                pyautogui.scroll(-50)  # scroll down
                cv2.putText(frame, 'Scroll Down', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Display
    cv2.imshow('Hand Gesture Control', frame)

    # Quit on 'q'
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
