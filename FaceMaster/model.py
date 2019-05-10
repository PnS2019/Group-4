from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout, AveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

data_pixels = np.genfromtxt('data/data_pixels.csv', dtype='float32', delimiter=',')
data_pixels = data_pixels.reshape((data_pixels.shape[0], 80, 80, 1))
data_pixels /= 255.
print("pixels loaded")
print(data_pixels.shape)
data_labels = np.genfromtxt('data/data_labels.csv', dtype='float32')
print(data_labels.shape)
print("labels loaded")
num_classes = 8

# shuffle and split
data_pixels, data_labels = shuffle(data_pixels, data_labels, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(
    data_pixels, data_labels, test_size=0.2, random_state=42)
print("separation done")
print(X_train.shape)
print(X_test.shape)

print(X_train[1].flatten())
print(X_test[1].flatten())

plt.figure()
plt.hist(y_train.flatten())
plt.show()

# converting the input class labels to categorical labels for training
Y_train = to_categorical(y_train, num_classes=num_classes)
Y_test = to_categorical(y_test, num_classes=num_classes)

# define a model
num_train_samples = X_train.shape[0]
num_test_samples = X_test.shape[0]
input_shape = X_train.shape[1:]
print(num_train_samples)
print(num_test_samples)
print(input_shape)

kernel_sizes = [(3, 3), (3, 3), (3, 3), (3, 3)]
num_kernels = [32, 16, 16, 16]

pool_sizes = [(3, 3), (3, 3), (3, 3)]
pool_strides = [(2, 2), (2, 2), (2, 2)]

num_hidden_units = [128, 64]

x = Input(shape=input_shape)
y = Conv2D(num_kernels[0], kernel_sizes[0], activation='relu', kernel_initializer="he_normal")(x)
y = BatchNormalization()(y)
y = Conv2D(num_kernels[1], kernel_sizes[1], activation='relu', kernel_initializer="he_normal")(y)
y = BatchNormalization()(y)
y = AveragePooling2D(pool_sizes[0], pool_strides[0])(y)

y = Conv2D(num_kernels[2], kernel_sizes[2], activation='relu', kernel_initializer="he_normal")(y)
y = BatchNormalization()(y)
# y = Conv2D(num_kernels[3], kernel_sizes[3], activation='relu', kernel_initializer="he_normal")(y)
# y = BatchNormalization()(y)
y = AveragePooling2D(pool_sizes[1], pool_strides[1])(y)

y = Flatten()(y)
y = Dense(num_hidden_units[0], activation='relu', kernel_initializer="he_normal")(y)
y = Dropout(rate=0.2)(y)
y = Dense(num_hidden_units[1], activation='relu', kernel_initializer="he_normal")(y)
y = Dropout(rate=0.2)(y)
y = Dense(num_classes, activation='softmax', kernel_initializer="he_normal")(y)
model = Model(x, y)

# print model summary
model.summary()

# compile the model against the binary cross entropy loss
# and use SGD optimizer
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy",
              optimizer=sgd,
              metrics=["accuracy"])

history = model.fit(
    x=X_train, y=Y_train,
    batch_size=64, epochs=25,
    validation_data=(X_test, Y_test))


# datagen = ImageDataGenerator(
#     featurewise_center=False,
#     featurewise_std_normalization=False,
#     rotation_range=0,
#     width_shift_range=0,
#     height_shift_range=0,
#     horizontal_flip=True,
#     vertical_flip=False)
#
# datagen.fit(X_train)
# model.fit_generator(datagen.flow(X_train, Y_train, batch_size=64),
#                     steps_per_epoch=100, epochs=20,
#                     validation_data=(X_test, Y_test))

train_score = model.evaluate(X_train, Y_train, verbose=0)
print('Train loss: ', train_score[0])
print('Train accuracy: ', 100 * train_score[1])

test_score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss: ', test_score[0])
print('Test accuracy: ', 100 * test_score[1])

# save the model
model.save("model/facial_expression_detection3.hdf5")
