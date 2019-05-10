import numpy as np

ck_pixels = np.genfromtxt('ck_data_pixels.csv', dtype='float32', delimiter=',')
ck_labels = np.genfromtxt('ck_data_labels.csv', dtype='float32')
jaffe_pixels = np.genfromtxt('jaffe_data_pixels.csv', dtype='float32', delimiter=',')
jaffe_labels = np.genfromtxt('jaffe_data_labels.csv', dtype='float32')

data_pixels = np.vstack((ck_pixels, jaffe_pixels))
data_labels = np.append(ck_labels, jaffe_labels)

print("pixels shape: ", data_pixels.shape)
print("labels shape; ", len(data_labels))

np.savetxt('data_pixels.csv', data_pixels, fmt='%i', delimiter=',')
np.savetxt('data_labels.csv', data_labels, fmt='%i')
