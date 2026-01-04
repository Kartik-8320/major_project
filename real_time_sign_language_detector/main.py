import cv2
import numpy as np
import mediapipe as mp
import pyttsx3
import tensorflow as tf
from tensorflow.keras.models import load_model
from collections import deque
import time
import json
import threading
from queue import Queue

# ================= CONFIG =================
IMG_SIZE = 64
MODEL_PATH = "models/sign_language_cnn_tf213.keras"
CLASS_NAMES_PATH = "models/class_names.json"

CONFIDENCE_THRESHOLD = 0.75
SPEECH_DELAY = 2.0          # seconds
PREDICTION_WINDOW = 15      # frames
# ==========================================

# Load class names
with open(CLASS_NAMES_PATH, "r") as f:
    class_dict = json.load(f)
CLASS_NAMES = list(class_dict.keys())

# Load CNN model
model = load_model(MODEL_PATH, compile=False)

# ================= TEXT TO SPEECH =================
engine = pyttsx3.init()
engine.setProperty("rate", 150)

speech_queue = Queue()

def speech_worker():
    while True:
        text = speech_queue.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()
        speech_queue.task_done()

speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()
# ==================================================

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

# Prediction smoothing
prediction_queue = deque(maxlen=PREDICTION_WINDOW)

# Speech control
last_spoken = ""
last_spoken_time = 0

print("🎥 Real-time Sign Language Detector Started")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            # Bounding box from landmarks
            x_list = [lm.x for lm in hand_landmarks.landmark]
            y_list = [lm.y for lm in hand_landmarks.landmark]

            padding = 40
            x_min = int(min(x_list) * w) - padding
            y_min = int(min(y_list) * h) - padding
            x_max = int(max(x_list) * w) + padding
            y_max = int(max(y_list) * h) + padding

            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w, x_max)
            y_max = min(h, y_max)

            hand_img = frame[y_min:y_max, x_min:x_max]

            if hand_img.size != 0:
                # Preprocess for CNN
                hand_img = cv2.resize(hand_img, (IMG_SIZE, IMG_SIZE))
                hand_img = hand_img / 255.0
                hand_img = np.expand_dims(hand_img, axis=0)

                # Predict
                preds = model.predict(hand_img, verbose=0)
                prediction_queue.append(preds[0])

                avg_preds = np.mean(prediction_queue, axis=0)
                class_id = np.argmax(avg_preds)
                confidence = avg_preds[class_id]
                label = CLASS_NAMES[class_id]

                # Draw bounding box and label
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{label} ({confidence:.2f})",
                    (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )

                # Speech trigger (SAFE)
                current_time = time.time()
                if confidence > CONFIDENCE_THRESHOLD:
                    if (label != last_spoken) or (current_time - last_spoken_time > SPEECH_DELAY):
                        print("🔊 Speaking:", label)
                        speech_queue.put(label)
                        last_spoken = label
                        last_spoken_time = current_time

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Real-Time Sign Language Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Clean shutdown
speech_queue.put(None)
engine.stop()
cap.release()
cv2.destroyAllWindows()
