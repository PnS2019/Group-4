from sys import path
path.append("/Users/xiaorui/Desktop/Group-4/FaceMaster/")
from face_detect import get_face
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt

model = load_model('../model/facial_expression_detection3.hdf5')
img3 = cv2.imread('me3.png')

face3 = get_face(img3).reshape(1, 80, 80, 1)/255.

emotions = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise', 'natural']

pred = model.predict(face3)[0]
emotion = emotions[int(np.argmax(model.predict(face3), axis=1).astype(np.int))]

x = [2 * i for i in range(8)]
plt.bar(x=x, height=pred, width=0.8, alpha=0.8, color='red', label="Prediction")
plt.ylim(0, 1)
plt.ylabel("Probability")
plt.xticks([i for i in x], emotions)
plt.xlabel("Emotion")
plt.title("Your emotion is " + emotion + "!")
plt.legend()
plt.show()
plt.waitforbuttonpress(0)
plt.close
