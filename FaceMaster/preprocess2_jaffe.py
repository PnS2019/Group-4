import os
import cv2
import numpy as np
from utils import get_face

path = 'jaffe'
jaffe_data_pixels = []
jaffe_data_labels = []

image_names = os.listdir(path)

for image_name in image_names:
    try:
        image_path = os.path.join(path, image_name)
        image = cv2.imread(image_path)
        print(image_name)
        face = get_face(image)
        if face != []:
            face = face.flatten()
            if jaffe_data_pixels == []:
                jaffe_data_pixels = face
            else:
                jaffe_data_pixels = np.vstack((jaffe_data_pixels, face))
            print(face)
            print("data_pixels's shape: ", jaffe_data_pixels.shape)

            label = image_name[3:5]
            if label == 'AN':
                jaffe_data_labels.append(0)
            elif label == 'DI':
                jaffe_data_labels.append(2)
            elif label == 'FE':
                jaffe_data_labels.append(3)
            elif label == 'HA':
                jaffe_data_labels.append(4)
            elif label == 'SA':
                jaffe_data_labels.append(5)
            elif label == 'SU':
                jaffe_data_labels.append(6)
            elif label == 'NE':
                jaffe_data_labels.append(7)
            print("number of labes: ", len(jaffe_data_labels))
            print("\n\n")
    except Exception as e:
        print(e)
        continue

np.savetxt('data/jaffe_data_pixels.csv', jaffe_data_pixels, fmt='%i', delimiter=',')
np.savetxt('data/jaffe_data_labels.csv', jaffe_data_labels, fmt='%i')
