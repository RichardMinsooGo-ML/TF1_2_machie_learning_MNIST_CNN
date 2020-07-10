import numpy as np
from random import randint
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils

np.random.seed(1234)

batch_size = 128
num_classes = 10
epochs = 8

# input image dimensions
img_rows, img_cols = 28, 28

def generator(features, labels, batch_size):

    # create empty arrays to contain batch of features and labels
    batch_features = np.zeros((batch_size, img_rows, img_cols, 1))
    batch_labels = np.zeros((batch_size, 10))

    while True:
        for i in range(batch_size):
            # choose random index in features
            index = np.random.randint(0, len(features)-1)
            random_augmented_image, random_augmented_label = features[index], labels[index]
            batch_features[i] = random_augmented_image
            batch_labels[i] = random_augmented_label

        yield batch_features, batch_labels

# load pre-shuffled MNIST data
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# pre-process data
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(Y_train, 10)
Y_test = np_utils.to_categorical(Y_test, 10)
print(Y_train.shape)

# build the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.fit(X_train, Y_train, batch_size=32, epochs=10, verbose=1)
model.fit_generator(generator(X_train, Y_train, 32), steps_per_epoch=X_train.shape[0] / 32, epochs=10, verbose=1)
