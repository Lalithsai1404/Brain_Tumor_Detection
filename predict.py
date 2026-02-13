# predict.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import sys
import os   # 👈 Added this line

MODEL_PATH = 'model/brain_tumor_model.h5'
IMG_SIZE = (224, 224)

model = load_model(MODEL_PATH)

# Provide image path as command-line argument
if len(sys.argv) < 2:
    print("Usage: python predict.py path/to/image.jpg")
    sys.exit()

img_path = sys.argv[1]
img = cv2.imread(img_path)
if img is None:
    print("Image not found!")
    sys.exit()

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img, IMG_SIZE)
img_array = img_resized / 255.0
img_array = np.expand_dims(img_array, axis=0)

pred = model.predict(img_array)
class_idx = np.argmax(pred)
class_labels = sorted(os.listdir('data/Training'))  # ensures same order each time
print("Prediction:", class_labels[class_idx])
print("Confidence:", pred[0][class_idx])
