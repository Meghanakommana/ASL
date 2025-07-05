import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 📥 Load the dataset
df = pd.read_csv("hand_sign_data.csv")

# 🧠 Separate features and labels
X = df.iloc[:, :-1].values    # 63 landmark features
y = df.iloc[:, -1].values     # Labels (A–K)

# 🔢 Encode labels (A–K → 0–10)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 🧪 Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 🤖 Train MLP model
model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
model.fit(X_train, y_train)

# 📊 Evaluate the model
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# 💾 Save model and label encoder
joblib.dump(model, "hand_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("✅ Model and label encoder saved as 'hand_model.pkl' and 'label_encoder.pkl'")
