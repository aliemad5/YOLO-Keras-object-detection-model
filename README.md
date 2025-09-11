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
Install dependencies with:
```bash
pip install -r requirements.txt

---


```python
from ultralytics import YOLO       # YOLO object detection
from tensorflow import keras       # Keras deep learning framework
import cv2                         # OpenCV for computer vision
import numpy as np                 # NumPy for array operations
