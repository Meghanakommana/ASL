import cv2
import mediapipe as mp
import csv
import os
import time

# 👋 A–Z (excluding 'J' if needed)
labels = [chr(i) for i in range(65, 91)]  # ['A', ..., 'Z']

# Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# CSV setup
filename = "hand_sign_data.csv"
file_exists = os.path.isfile(filename)

csv_file = open(filename, mode='a', newline='')
csv_writer = csv.writer(csv_file)

# Write header once
if not file_exists:
    header = [f'{axis}{i}' for i in range(21) for axis in ['x', 'y', 'z']] + ['label']
    csv_writer.writerow(header)

# Webcam
cap = cv2.VideoCapture(0)

frames_per_letter = 60  # Aim for 60 samples per letter
print("🚀 Starting auto data collection... Press 'q' to quit anytime.\n")

for label in labels:
    print(f"✋ Show the sign for: {label}")
    time.sleep(2)  # 2-second pause for you to prepare

    count = 0
    while count < frames_per_letter:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for lm in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
                data = []
                for point in lm.landmark:
                    data.extend([point.x, point.y, point.z])

                if len(data) == 63:
                    row = data + [label]
                    csv_writer.writerow(row)
                    count += 1
                    print(f"✅ Collected {count}/{frames_per_letter} for {label}", end='\r')

        cv2.putText(frame, f"Sign: {label} ({count}/{frames_per_letter})", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("Data Collection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            csv_file.close()
            cv2.destroyAllWindows()
            print("\n🚪 Exiting early, data saved.")
            exit()

print("\n✅ Dataset collection complete for A–Z!")
cap.release()
csv_file.close()
cv2.destroyAllWindows()
