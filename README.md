# YOLO-Keras-object-detection-model
"""
YOLO + Keras Integration for Real-Time Video Classification

This script:
1. Uses a YOLOv11n model (from Ultralytics) to detect objects in real-time.
2. Crops detected objects and passes them to a custom Keras model for classification.
3. Displays predictions with bounding boxes and labels on the live webcam feed.

Requirements:
- ultralytics
- tensorflow
- opencv-python
- numpy
"""

from ultralytics import YOLO
from tensorflow import keras
import cv2
import numpy as np


# Load YOLO model (pre-trained weights)
yolo_model = YOLO("yolov11n.pt")

# Load custom Keras model
keras_model = keras.models.load_model("mykeras.h5")

# Start webcam
video = cv2.VideoCapture(0)

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    # Run YOLO prediction
    results = yolo_model.predict(frame, verbose=False)

    # Extract bounding boxes
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

    # Process each detection
    for x1, y1, x2, y2 in boxes:
        # Crop and preprocess image for Keras model
        cropped = frame[y1:y2, x1:x2]
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(cropped_rgb, (512, 512))
        expanded = np.expand_dims(resized, axis=0)

        # Run Keras prediction
        prediction = keras_model.predict(expanded, verbose=0)
        class_id = np.argmax(prediction)

        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 3)
        cv2.putText(frame, f"Class: {class_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Show video output
    cv2.imshow("YOLO + Keras Project", frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
