from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np

data_pixels = np.genfromtxt('data_pixels2.csv', dtype='float64', delimiter=',')
data_pixels = data_pixels.reshape((data_pixels.shape[0], 80, 80, 1))
print("pixels loaded")
print(data_pixels.shape)
data_labels = np.genfromtxt('data_labels2.csv', dtype='int64')
print(data_labels.shape)
print("labels loaded")
num_classes = 6

# shuffle and split
data_pixels, data_labels = shuffle(data_pixels, data_labels, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(
    data_pixels, data_labels, test_size=0.1, random_state=42)
print("separation done")
print(X_train.shape)
print(X_test.shape)

# converting the input class labels to categorical labels for training
train_Y = to_categorical(y_train, num_classes=num_classes)
test_Y = to_categorical(y_test, num_classes=num_classes)
print(train_Y.shape)
print(test_Y.shape)

# define a model
num_train_samples = X_train.shape[0]
num_test_samples = X_test.shape[0]
input_shape = X_train.shape[1:]
print(num_train_samples)
print(num_test_samples)
print(input_shape)

kernel_sizes = [(7, 7), (5, 5)]
num_kernels = [20, 25]

pool_sizes = [(2, 2), (2, 2)]
pool_strides = [(2, 2), (2, 2)]

num_hidden_units = 200

x = Input(shape=input_shape)
y = Conv2D(num_kernels[0], kernel_sizes[0], activation='relu')(x)
y = MaxPooling2D(pool_sizes[0], pool_strides[0])(y)
y = Conv2D(num_kernels[1], kernel_sizes[1], activation='relu')(y)
y = MaxPooling2D(pool_sizes[1], pool_strides[1])(y)
y = Flatten()(y)
y = Dense(num_hidden_units, activation='relu')(y)
y = Dense(num_classes, activation='softmax')(y)
model = Model(x, y)

# print model summary
model.summary()

# compile the model aganist the binary cross entropy loss
# and use SGD optimizer
model.compile(loss="categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=False,
    rotation_range=0,
    width_shift_range=0,
    height_shift_range=0,
    horizontal_flip=True,
    vertical_flip=False)

datagen.fit(X_train)
datagen.standardize(X_test)
model.fit_generator(datagen.flow(X_train, train_Y, batch_size=32),
                    steps_per_epoch=len(X_train) / 32, epochs=20,
                    validation_data=(X_test,test_Y))

model.save("facial_expression_detection.hdf5")