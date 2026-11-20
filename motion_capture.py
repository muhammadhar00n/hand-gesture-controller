import cv2
import mediapipe as mp
from pynput.keyboard import Controller as KBController
from pynput.mouse import Controller as MouseController, Button
import time

# ─── Setup ─────────────────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

kb = KBController()
mouse = MouseController()

# Timing to avoid repeats
last_action_time = 0
action_cooldown = 0.5  # seconds
aiming = False

# ─── Finger Counting Logic ─────────────────────────────────────────────────────
FINGER_TIPS = [
    mp_hands.HandLandmark.THUMB_TIP,
    mp_hands.HandLandmark.INDEX_FINGER_TIP,
    mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
    mp_hands.HandLandmark.RING_FINGER_TIP,
    mp_hands.HandLandmark.PINKY_TIP,
]
FINGER_PIPS = [
    mp_hands.HandLandmark.THUMB_IP,
    mp_hands.HandLandmark.INDEX_FINGER_PIP,
    mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
    mp_hands.HandLandmark.RING_FINGER_PIP,
    mp_hands.HandLandmark.PINKY_PIP,
]

def count_extended_fingers(landmarks):
    """Return how many fingers are extended. More strict on thumb."""
    count = 0
    for tip, pip in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
        if landmarks[tip].y < landmarks[pip].y:
            count += 1

    thumb_tip = landmarks[FINGER_TIPS[0]]
    thumb_ip = landmarks[FINGER_PIPS[0]]
    wrist = landmarks[mp_hands.HandLandmark.WRIST]

    if abs(thumb_tip.y - wrist.y) < 0.1 and abs(thumb_tip.x - thumb_ip.x) > 0.05:
        count += 1

    return count

# ─── Action Triggers ───────────────────────────────────────────────────────────
def try_fire_or_select(finger_count):
    global last_action_time, aiming
    now = time.time()
    if now - last_action_time < action_cooldown:
        return
    last_action_time = now

    if finger_count == 0:
        if not aiming:
            mouse.press(Button.right)
            aiming = True
            print("[AIM] Hold Right-Click")
        mouse.click(Button.left)
        print("[FIRE] Left-Click")
    else:
        if aiming:
            mouse.release(Button.right)
            aiming = False
            print("[AIM] Released Right-Click")

        if 1 <= finger_count <= 5:
            key = str(finger_count)
            kb.press(key)
            kb.release(key)
            print(f"[ACTION] Weapon Slot {finger_count} → Key {key}")

# ─── Main Loop ─────────────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                lm = results.multi_hand_landmarks[0].landmark
                cnt = count_extended_fingers(lm)
                try_fire_or_select(cnt)

                mp.solutions.drawing_utils.draw_landmarks(
                    frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS
                )

                cv2.putText(frame, f"Fingers: {cnt}", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            cv2.imshow("Hand Control (Q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        if aiming:
            mouse.release(Button.right)
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
