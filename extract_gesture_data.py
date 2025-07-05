import cv2
import mediapipe as mp
import os
import pandas as pd

# 📁 Folder with your recorded videos
video_folder = "videos"

# 🧠 Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# 📊 For collecting landmarks
all_data = []
all_labels = []

# Loop through all video files
for filename in os.listdir(video_folder):
    if filename.endswith(".mp4"):
        label = filename.split(".")[0]  # 'excite.mp4' → 'excite'
        path = os.path.join(video_folder, filename)

        cap = cv2.VideoCapture(path)
        print(f"🔍 Processing {filename}...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(image)

            if result.multi_hand_landmarks:
                for lm in result.multi_hand_landmarks:
                    landmarks = []
                    for point in lm.landmark:
                        landmarks.extend([point.x, point.y, point.z])

                    if len(landmarks) == 63:
                        all_data.append(landmarks)
                        all_labels.append(label)

        cap.release()

# 🧾 Save landmarks to CSV
df = pd.DataFrame(all_data)
df["label"] = all_labels
df.to_csv("gesture_data.csv", index=False)
print("✅ Extracted data saved as gesture_data.csv")
