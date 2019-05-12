from tensorflow.python.keras.models import load_model
import numpy as np
import cv2

model = load_model("facial_expression_detection2.hdf5")
emotions = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
def predict(img):
	pred = np.argmax(model.predict(img), axis=1).astype(np.int)
	print(emotions[int(pred)])
