import os
import cv2
import numpy as np
from utils import get_face

data_image_path = 'cohn-kanade-images'
data_label_path = 'Emotion'
files = os.listdir(data_label_path)

data_pixels = []
data_labels = []

num_undetectable_faces = 0


def read_image(path):
    global data_pixels, data_labels, num_undetectable_faces
    print("the path of image: ", path)
    img = cv2.imread(path)
    face = get_face(img)
    if face == []:
        print("no face detected or not clear")
        del data_labels[len(data_labels)]
        num_undetectable_faces += 1
    else:
        face = face.flatten()
        if data_pixels == []:
            data_pixels = face
        else:
            data_pixels = np.vstack((data_pixels, face))
        print(face)
        print("data_pixels's shape: ", data_pixels.shape)


def read_label(path):
    global data_labels, tem
    print("the path of the label: ", path)
    try:
        label_file = open(path)
        label = float(label_file.read()) - 1
        if label == None:
            tem = 1
        else:
            data_labels.append(label)
            print(label)
            print("number of labels: ", len(data_labels))
    except Exception as e:
        tem = 1
        print(repr(e))


for file in files:
    file_path = os.path.join(data_label_path, file)
    subfiles = os.listdir(file_path)
    for subfile in subfiles:
        subfile_path = os.path.join(file_path, subfile)
        tem = 0
        file_name = os.listdir(subfile_path)
        if file_name != []:
            file_name = file_name[0]
            label_path = os.path.join(subfile_path, file_name)
            read_label(label_path)
            if tem is not 1:
                image_name = file_name[:17]+'.png'
                image_path = os.path.join(data_image_path, file, subfile, image_name)
                read_image(image_path)
            print("\n\n")
        else:
            continue


print(data_pixels)
print("Summery: the shape of pixels: ", data_pixels.shape)
print(data_labels)
print("Summery: the shape of labels: ", len(data_labels))
print("the number of undetectable faces: ", num_undetectable_faces)

np.savetxt('ck_data_pixels.csv', data_pixels, fmt='%i', delimiter=',')
np.savetxt('ck_data_labels.csv', data_labels, fmt='%i')

