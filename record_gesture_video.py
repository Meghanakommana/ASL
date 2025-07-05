import cv2
import os
import json

# 🎯 Load action labels
with open("action_labels.json") as f:
    label_map = json.load(f)

labels = list(label_map.keys())
video_dir = "videos"
os.makedirs(video_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

print("🎥 Press 'r' to start/stop recording, 'n' to move to next label, 'q' to quit")

recording = False
out = None
current_label_index = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    label = labels[current_label_index]
    cv2.putText(frame, f"Label: {label} | Recording: {recording}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if recording else (0, 255, 0), 2)
    cv2.imshow("Video Recorder", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif key == ord("n"):
        if recording and out:
            out.release()
        current_label_index = (current_label_index + 1) % len(labels)
        recording = False
        print(f"➡️ Switched to: {labels[current_label_index]}")
    elif key == ord("r"):
        if not recording:
            path = os.path.join(video_dir, f"{label}.mp4")
            out = cv2.VideoWriter(path, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
            recording = True
            print(f"🎬 Recording started for: {label}")
        else:
            recording = False
            out.release()
            print(f"🛑 Recording stopped for: {label}")

    if recording and out:
        out.write(frame)

cap.release()
cv2.destroyAllWindows()
