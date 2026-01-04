import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

DATA_DIR = "data/raw_landmarks"
OUTPUT_DIR = "data/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

X, y = [], []

for gesture in os.listdir(DATA_DIR):
    gesture_dir = os.path.join(DATA_DIR, gesture)

    for file in os.listdir(gesture_dir):
        data = np.load(os.path.join(gesture_dir, file))
        X.append(data)
        y.append(gesture)

X = np.array(X)
y = np.array(y)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
np.save(os.path.join(OUTPUT_DIR, "y.npy"), y_encoded)
np.save(os.path.join(OUTPUT_DIR, "labels.npy"), encoder.classes_)

print("✅ Dataset ready")
print("Samples:", X.shape[0])
print("Features:", X.shape[1])
