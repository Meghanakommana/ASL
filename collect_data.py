import cv2
import mediapipe as mp
import pandas as pd
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
data = []
labels = []

print("📸 Show a sign (like A), press that key, repeat 50–100 times")
print("🚪 Press 'q' to quit and save the data")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract 21 hand landmarks (x, y, z)
            landmark_list = []
            for lm in hand_landmarks.landmark:
                landmark_list.extend([lm.x, lm.y, lm.z])

            key = cv2.waitKey(1) & 0xFF
            if 65 <= key <= 90:  # A-Z
                label = chr(key)
                data.append(landmark_list)
                labels.append(label)
                print(f"✅ Captured frame for label: {label}")
            elif key == ord('q'):
                print("🚪 Exiting and saving data.")
                cap.release()
                cv2.destroyAllWindows()
                df = pd.DataFrame(data)
                df['label'] = labels
                df.to_csv("hand_sign_data.csv", index=False)
                exit()

    cv2.imshow("Collect Hand Sign Data", frame)
