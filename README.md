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

## load custom model

This project uses a pre-trained Keras model saved as **`mykeras.h5`**.  
The full training code, data preprocessing, and experiments for this model are explained in another repository:   [Custom Keras Model Repo](https://github.com/YOUR-USERNAME/keras-model-repo) 
```python
keras_model = keras.models.load_model("mykeras.h5")
```


# Open Webcam 
video = cv2.VideoCapture(0)






