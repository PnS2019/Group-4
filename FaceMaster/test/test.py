import cv2
import numpy as np
from utils import get_face
from tensorflow.keras.models import load_model

model = load_model('facial_expression_detection.hdf5')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow('frame',frame)
    try:
        face = get_face(frame)
        face = face.reshape(1, 80, 80, 1)
        print(model.predict(face))
    except:
        continue
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()