

## Project Details

- **Creator:** Ali Emad Elsamanoudy 
- **Location:** Saudi Arabia
- **Phone** +966 59 645 2087
- **Date:** September 2025  
- **Description:**  
  This project combines **YOLOv11** object detection with a **custom Keras/TensorFlow classifier**  
  for real-time image classification through webcam input.  


# YOLO + Keras Real-Time Classification


This project combines **YOLO (Ultralytics)** for object detection and a custom **Keras model** for classification.  
It uses OpenCV to process webcam frames, detect objects, crop them, classify with Keras, and display results in real-time.  

---

## Features
- Real-time object detection with YOLOv11n  
- Crops detected objects and classifies them with a trained Keras model  
- Shows bounding boxes and predicted labels on the live webcam feed  

---
## Requirements



# YOLO + Keras Real-Time Classification

This project combines **YOLO (Ultralytics)** for object detection and a custom **Keras/TensorFlow model** for classification.  
It uses OpenCV to process webcam frames, detect objects, crop them, classify with Keras, and display results in real-time.  

---

## Imports
```python
from ultralytics import YOLO
from tensorflow import keras
import cv2
import numpy as np
```

## Load yolo model
```python
yolo_model = YOLO("yolov11n.pt")
```

## Load custom model

This project uses a pre-trained Keras model saved as **`mykeras.h5`**.  
The full training code, data preprocessing, and experiments for this model are explained in another repository:   [Custom Keras Model Repo](https://github.com/YOUR-USERNAME/keras-model-repo) 
```python
keras_model = keras.models.load_model("mykeras.h5")
```


# Open Webcam 
```python
video = cv2.VideoCapture(0)
```

## Main loop
```python
while video.isOpened():
    # Capture frame from webcam
    ret, frame = video.read()
    if not ret:
        break

    # Run YOLO object detection
    results = yolo_model.predict(frame, verbose=False)

    # Extract bounding boxes
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

    # Process detections if any
    if len(boxes) > 0:
        for x1, y1, x2, y2 in boxes:
            # Crop detected region
            cropped = frame[y1:y2, x1:x2]

            # Preprocess for Keras model
            cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(cropped_rgb, (512, 512))
            expanded = np.expand_dims(resized, axis=0)

            # Run prediction with Keras model
            prediction = kmodel.predict(expanded, verbose=0)
            class_id = np.argmax(prediction)

            # Draw bounding box and prediction
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 3)
            cv2.putText(frame,
                        f"Class: {class_id}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 0),
                        2)

    # Show output frame
    cv2.imshow("YOLO + Keras Object Detection", frame)

```
## Show output and exit
```python
    # Display webcam feed with detections
    cv2.imshow("YOLO + Keras Project", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
video.release()
cv2.destroyAllWindows()
```

    

