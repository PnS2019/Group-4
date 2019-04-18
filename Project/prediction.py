from tensorflow.keras.models import load_model
import numpy as np
import cv2

model = load_model("facial_expression_detection.df5")

def predict(img):
    pred = model.predict(img)
    print(pred)