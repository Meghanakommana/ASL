import cv2
import numpy as np
import joblib
import mediapipe as mp
import os

# 🎛️ Mode: switch between 'letter' and 'action'
mode = 'letter'

# 📥 Load both models and encoders
letter_model = joblib.load("hand_model.pkl")
letter_encoder = joblib.load("label_encoder.pkl")

gesture_model = joblib.load("gesture_model.pkl")
gesture_encoder = joblib.load("gesture_encoder.pkl")

# 🧠 Gesture-to-sentence mapping
gesture_to_sentence = {
    "i":"I",
    "You":"You",
    "what":"what",
    "hello": "Hello",
    "I am":"I Am",
    "happy":"HAPPY",
    "excite": "I am excited",
    "what is up": "What's up",
    "name": "name",
    "love": "I love you",
    "beautiful": "look beautiful",
    "ugly": "look ugly",
    "How are you":"How are you",
    "how":"how",
    "when":"when",
}

# 🖐️ MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# 🎥 Start camera
cap = cv2.VideoCapture(0)
last_prediction = ""

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Flip + RGB
    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    prediction = ""

    if results.multi_hand_landmarks:
        for lm in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

            data = []
            for point in lm.landmark:
                data.extend([point.x, point.y, point.z])

            X_input = np.array(data).reshape(1, -1)

            if mode == 'letter':
                pred = letter_model.predict(X_input)[0]
                prediction = letter_encoder.inverse_transform([pred])[0]
            else:
                pred = gesture_model.predict(X_input)[0]
                gesture = gesture_encoder.inverse_transform([pred])[0]
                prediction = gesture_to_sentence.get(gesture, gesture)

            # Speak only when prediction changes
            if prediction != last_prediction:
                print(f"🔤 Prediction: {prediction}")
                last_prediction = prediction

    # Display mode & prediction
    cv2.putText(frame, f"Mode: {mode.upper()}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.putText(frame, f"{prediction}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow("Sign Language Interpreter", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif key == ord("m"):
        mode = "action" if mode == "letter" else "letter"

cap.release()
cv2.destroyAllWindows()
