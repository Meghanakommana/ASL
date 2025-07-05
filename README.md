# 🧠 Sign Language Interpreter 🤟 | Real-Time Gesture & action to Text & Action Conversion

> Built with by Meghana Kommana  
> Empowering communication through AI + Computer Vision.

---

## 🚀 Project Overview

This Sign Language Interpreter is a **real-time gesture recognition system** that detects hand gestures using a webcam and interprets them into:
- 🔤 **Alphabets (A–K)** – trained using hand landmarks
- 💬 **Custom actions** – like “I love you” or “What’s up?” using full-hand gestures

This project solves the communication gap between hearing and speech-impaired individuals and others by **translating hand signs into letters or meaningful sentences** using machine learning and computer vision.

---

## 🎯 Key Features

- ✅ Real-time hand detection with **MediaPipe**
- 🎥 Webcam-based input
- 🧠 Machine learning model using **scikit-learn (MLPClassifier)**
- 🪄 Predicts both **letter signs** and **custom action gestures**
- 🔁 Switch modes (letter ↔ action) by pressing `M`
- 📸 Custom gesture data collection using your webcam
- 🗣️ (Optional) Speak predictions out loud with `os.system("say")`
- 📂 Neatly organized training, prediction, and data collection files

---

## 🛠️ Tech Stack

| Layer            | Tech Used                          |
|------------------|------------------------------------|
| ✋ Hand Detection | `MediaPipe`, `OpenCV`              |
| 🧠 ML Model       | `scikit-learn`, `joblib`, `numpy`  |
| 📁 Data Handling  | `pandas`, `CSV`, `LabelEncoder`    |
| 🎤 Speech (opt.)  | macOS `say` command                |
| 📦 Packaging      | Python 3.9+, Virtualenv            |

---

