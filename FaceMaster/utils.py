import os
import cv2


def get_haarcascade_path(xml_name):
    """Get haar cascade path according to the given xml file."""
    return os.path.join('haarcascades', xml_name)


def get_face(img):
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    img = img.astype('uint8')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    (x, y, w, h) = faces[0]
    face = img[y:y+h, x:x+w]
    face = cv2.resize(face, (48, 48), interpolation=cv2.INTER_CUBIC)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    return face
