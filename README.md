# YOLO + Keras Real-Time Classification

This project combines **YOLO (Ultralytics)** for object detection and a custom **Keras model** for classification.  
It uses OpenCV to process webcam frames, detect objects, crop them, classify with Keras, and display results in real-time.  

---

## Features
- Real-time object detection with YOLOv11n
- Crops detected objects and runs them through a trained Keras model
- Shows bounding boxes and predicted labels on the live webcam feed

---

## Requirements
Install dependencies with:
```bash
pip install -r requirements.txt
"""
YOLO + Keras Integration for Real-Time Video Classification
Author: Ali Emad
"""

# --- Imports ---
from ultralytics import YOLO
from tensorflow import keras
import cv2
import numpy as np

# --- Load Models ---
# Load YOLOv11n model (pre-trained weights)
yolo_model = YOLO("yolov11n.pt")

# Load custom Keras model
keras_model = keras.models.load_model("mykeras.h5")

# --- Start Webcam ---
video = cv2.VideoCapture(0)

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    # --- YOLO Prediction ---
    results = yolo_model.predict(frame, verbose=False)

    # Extract bounding boxes from YOLO output
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

    # --- Process Each Detection ---
    for x1, y1, x2, y2 in boxes:
        # Crop detected region
        cropped = frame[y1:y2, x1:x2]

        # Convert BGR to RGB for Keras
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

        # Resize for Keras model input
        resized = cv2.resize(cropped_rgb, (512, 512))
        expanded = np.expand_dims(resized, axis=0)

        # --- Keras Prediction ---
        prediction = keras_model.predict(expanded, verbose=0)
        class_id = np.argmax(prediction)

        # --- Draw Results ---
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 3)
        cv2.putText(frame, f"Class: {class_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # --- Display Webcam Output ---
    cv2.imshow("YOLO + Keras Project", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --- Cleanup ---
video.release()
cv2.destroyAllWindows()
