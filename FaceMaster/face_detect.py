import cv2
from pnslib import utils


def get_face(img):
    face_cascade = cv2.CascadeClassifier(
        utils.get_haarcascade_path('haarcascade_frontalface_default.xml'))
        
    cv2.imshow('original image', img)
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=5)
    (x, y, w, h) = faces[0]
    face = img[y:y+h, x:x+w]
    
    face = cv2.resize(face, (48, 48), interpolation=cv2.INTER_CUBIC)
    
    cv2.imshow('face', face)
    
    return face
