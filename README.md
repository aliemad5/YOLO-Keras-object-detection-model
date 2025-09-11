"""
Train a custom CNN classifier with Keras/TensorFlow.
Dataset: images stored in folder structure -> imgpaths.csv + ylabels.csv
Author: Ali Emad Elsamanoudy
Date: September 2025
"""
## imports
```python
import tensorflow as tf
from keras.layers import Dense, Flatten, MaxPooling2D, Conv2D
from keras.models import Sequential
from keras.losses import SparseCategoricalCrossentropy
from keras.preprocessing import image
import pandas as pd
import numpy as np
```
# ============================
# Load Dataset
# ============================
```python
print("[INFO] Loading dataset from CSVs...")

y_train = pd.read_csv("ylabels.csv")       
x_images = pd.read_csv("imgpaths.csv")     

x_train = []
for img_path in x_images["path"]:
    img = image.load_img(img_path, target_size=(512, 512))
    img = image.img_to_array(img) / 255.0
    x_train.append(img)

x_train = np.array(x_train)
y_train = np.array(y_train).squeeze()

print(f"[INFO] Dataset loaded: {x_train.shape[0]} images, {len(np.unique(y_train))} classes")
```
# ============================
# Build Model
# ============================
```python
model = Sequential([
    Conv2D(32, (5, 5), padding="same", activation="relu", input_shape=(512, 512, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (4, 4), padding="same", activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation="relu"),
    Dense(50, activation="softmax")   
])

model.compile(optimizer="adam",
              loss=SparseCategoricalCrossentropy(),
              metrics=["accuracy"])

print("[INFO] Model compiled.")
```
# ============================
# Train
# ============================
```python
print("[INFO] Starting training...")
history = model.fit(
    x_train, y_train,
    batch_size=64,
    epochs=20,
    validation_split=0.2
)
```
# ============================
# Save Model
# ============================
```python
model.save("mykeras.h5")
print("[INFO] Model saved as mykeras.h5")
```
# ============================
# Save Class Mapping
# ============================
```python
classes = sorted(np.unique(y_train))
with open("class_mapping.txt", "w") as f:
    for idx, cls in enumerate(classes):
        f.write(f"{idx}: class_{cls}\n")

print("[INFO] class_mapping.txt created")
```
