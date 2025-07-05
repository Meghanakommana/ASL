import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 📥 Load gesture landmark data
df = pd.read_csv("gesture_data.csv")
X = df.iloc[:, :-1].values  # 63 landmark values
y = df["label"].values      # gesture label (sentence keyword)

# 🔠 Encode gesture labels (e.g. 'excite' → 0)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 🧪 Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 🧠 Train the model
model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
model.fit(X_train, y_train)

# 📊 Evaluate
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# 💾 Save model & label encoder
joblib.dump(model, "gesture_model.pkl")
joblib.dump(le, "gesture_encoder.pkl")
print("✅ Model and encoder saved as gesture_model.pkl & gesture_encoder.pkl")
