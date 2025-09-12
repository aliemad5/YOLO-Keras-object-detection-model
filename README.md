


Author: Ali Emad Elsamanoudy


Date: September 2025

## Imports
```python

from ultralytics import YOLO
from tensorflow import keras
import cv2
import numpy as np
```
## Load YOLO model
python
yolo_model = YOLO("yolov11n.pt")
## Load custom model
This project uses a pre-trained Keras model saved as **mykeras.h5**. The full training code, data preprocessing, and experiments for this model are explained in another repository: [Custom Keras Model Repo](https://github.com/YOUR-USERNAME/keras-model-repo)
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
    ret, frame = video.read()
    if not ret:
        break

    
    results = yolo_model.predict(frame, verbose=False)

    
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

    if len(boxes) > 0:
        for x1, y1, x2, y2 in boxes:
            # Crop detected region
            cropped = frame[y1:y2, x1:x2]

            cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(cropped_rgb, (512, 512))
            expanded = np.expand_dims(resized, axis=0)

            
            prediction = kmodel.predict(expanded, verbose=0)
            class_id = np.argmax(prediction)

            n
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 3)
            cv2.putText(frame,
                        f"Class: {class_id}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 0),
                        2)

```
## Show output and exit
```python
    cv2.imshow("YOLO + Keras Project", frame)

   
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


video.release()
cv2.destroyAllWindows()
```
