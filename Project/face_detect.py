import cv2
from pnslib import utils


def get_face(img):
    face_cascade = cv2.CascadeClassifier(
        utils.get_haarcascade_path('haarcascade_frontalface_default.xml'))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    (x, y, w, h) = faces[0]
    face = img[y:y+h, x:x+w]
    face = cv2.resize(face, (80, 80), interpolation=cv2.INTER_CUBIC)
    #face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    return face
