import cv2
import numpy as np
import os
import json

# 🔄 Load labels
with open("action_labels.json") as f:
    label_map = json.load(f)

labels = list(label_map.keys())
data_path = "action_data"
os.makedirs(data_path, exist_ok=True)

# 🎥 Start camera
cap = cv2.VideoCapture(0)
current_label = labels[0]
count = 0
total_per_label = 100  # Adjust as needed

print("Press 'n' to move to next label")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    display_text = f"Label: {current_label} | Count: {count}"
    cv2.putText(frame, display_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Collecting Gestures", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('n'):
        idx = labels.index(current_label)
        current_label = labels[(idx + 1) % len(labels)]
        count = 0
    elif key == ord('c'):
        save_path = os.path.join(data_path, f"{current_label}_{count}.jpg")
        cv2.imwrite(save_path, frame)
        print(f"✅ Saved: {save_path}")
        count += 1
        if count >= total_per_label:
            print(f"🎉 Collected {total_per_label} for {current_label}")
            idx = labels.index(current_label)
            current_label = labels[(idx + 1) % len(labels)]
            count = 0

cap.release()
cv2.destroyAllWindows()
