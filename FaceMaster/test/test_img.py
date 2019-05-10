from utils import get_face
from tensorflow.keras.models import load_model
import cv2
import numpy as np


model = load_model('/Users/xiaorui/Desktop/FaceMaster/model/facial_expression_detection3.hdf5')
img1 = cv2.imread('me1.png')
img2 = cv2.imread('me2.png')
img3 = cv2.imread('me3.png')

face1 = get_face(img1).reshape(1, 80, 80, 1)/255.
face2 = get_face(img2).reshape(1, 80, 80, 1)/255.
face3 = get_face(img3).reshape(1, 80, 80, 1)/255.

emotions = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise', 'natural']
print('im1')
print(emotions[int(np.argmax(model.predict(face1), axis=1).astype(np.int))])
print('im2')
print(emotions[int(np.argmax(model.predict(face2), axis=1).astype(np.int))])
print('im3')
print(emotions[int(np.argmax(model.predict(face3), axis=1).astype(np.int))])
