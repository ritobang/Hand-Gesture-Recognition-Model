import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def count_fingers(hand_landmarks):

    finger_tips = [4, 8, 12, 16, 20]
    finger_pips = [2, 6, 10, 14, 18]

    fingers_extended = []

    thumb_tip = hand_landmarks.landmark[finger_tips[0]]
    thumb_pip = hand_landmarks.landmark[finger_pips[0]]
    if thumb_tip.x < thumb_pip.x:
        fingers_extended.append(1)
    else:
        fingers_extended.append(0)

    for i in range(1, 5):
        tip = hand_landmarks.landmark[finger_tips[i]].y
        pip = hand_landmarks.landmark[finger_pips[i]].y
        fingers_extended.append(1 if tip < pip else 0)

    return fingers_extended


def classify_gesture(finger_states, hand_landmarks):

    if finger_states == [0, 0, 0, 0, 0]:
        return "Fist"
    elif finger_states == [1, 1, 1, 1, 1]:
        return "Stop"
    elif finger_states == [1, 0, 0, 0, 0]:
        return "Thumbs Up"
    elif finger_states == [0, 1, 1, 0, 0]:
        return "Peace"
    elif finger_states == [0, 1, 0, 0, 0]:
        return "Pointing"
    elif finger_states == [1, 0, 0, 0, 1]:
        return "Call Me"
    elif finger_states == [1, 1, 0, 0, 1]:
        return "ILY (I Love You)"
    elif finger_states == [0, 0, 1, 1, 1]:
        return "OK Sign"
    elif finger_states == [0, 0, 1, 0, 0]:
        return "Middle Finger"
    elif finger_states == [0, 0, 0, 1, 0]:
        return "Ring Finger"
    elif finger_states == [0, 0, 0, 0, 1]:
        return "Pinky Up"
    elif finger_states == [0, 1, 0, 0, 1]:
        return "Rock"
    elif finger_states == [1, 1, 1, 0, 0]:
        return "Gun"
    else:
        return "Unknown Gesture "

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8, max_num_hands=2) as hands:
    cv2.namedWindow("Hand Gesture Recognition", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Hand Gesture Recognition", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(rgb_frame)

        gesture_text = "No Hand Detected"

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                fingers = count_fingers(hand_landmarks)
                gesture_text = classify_gesture(fingers, hand_landmarks)

        cv2.putText(frame, f"Gesture: {gesture_text}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)

        cv2.imshow("Hand Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
