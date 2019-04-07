import csv
import cv2
import random
import numpy as np

import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Dropout, Cropping2D, Conv2D, BatchNormalization, GaussianNoise, GaussianDropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# hyperparameters
BATCH_SIZE = 128
STEERING_CORRECTION = 0.2

def preprocess_image(image):
    # Image Preprocessing (cropping and resize)

    def crop_image(img):
        height = img.shape[0]
        width = img.shape[1]
        crop_top = image[height-(height-70):height, 0:width]
        height = crop_top.shape[0]
        crop_bottom = crop_top[0:height-25, 0:width]
        return crop_bottom

    def resize_image(image):
        return cv2.resize(image, (200, 66))

    cropped_img = crop_image(image)
    resized_img = resize_image(cropped_img)
    result_img = resized_img

    return result_img

def load_training_data():
    # Loading data from driving_log.csv
    lines = []
    with open('driving_data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    # Import images and measurements
    images = []
    measurements = []
    for line in lines[1:]:

        # load images
        center_image = cv2.imread('driving_data/' + line[0])
        left_image = cv2.imread('driving_data/' + str(line[1]).replace(' ', ''))
        right_image = cv2.imread('driving_data/' + str(line[2]).replace(' ', ''))

        # read steering
        steering_center = float(line[3])
        steering_left = steering_center + STEERING_CORRECTION
        steering_right = steering_center - STEERING_CORRECTION

        # add center image
        images.append(preprocess_image(center_image))
        measurements.append(steering_center)

        # add left image
        images.append(preprocess_image(left_image))
        measurements.append(steering_left)

        # add right image
        images.append(preprocess_image(right_image))
        measurements.append(steering_right)

        # flip all that are not driving straight (50%)
        if abs(steering_center) > 0.1 and random.random() >= .5:
            flip_image = cv2.flip(center_image, 1)
            steering_flip = -steering_center
            images.append(preprocess_image(flip_image))
            measurements.append(steering_flip)

    return np.array(images), np.array(measurements)

def build_LeNet(X_train, y_train):
    model = Sequential()
    model.add(Convolution2D(6, 5, 5, activation='relu', input_shape=(66,200,3)))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dropout(0.2))
    model.add(Dense(84))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, validation_split=0.2, batch_size=BATCH_SIZE, shuffle=True, nb_epoch=1)
    return model

def build_NVIDIA(X_train, y_train):
    model = Sequential()

    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(66,200,3)))
    model.add(Convolution2D(24, 5, 5, border_mode="valid", subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(36, 5, 5, border_mode="valid", subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(48, 5, 5, border_mode="valid", subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(64, 3, 3, border_mode="valid", subsample=(1, 1), activation="relu"))
    model.add(Convolution2D(64, 3, 3, border_mode="valid", subsample=(1, 1), activation="relu"))
    model.add(Flatten())
    model.add(Dense(100, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    history_object = model.fit(X_train, y_train, validation_split=0.2, batch_size=BATCH_SIZE, shuffle=True, nb_epoch=3)

    return model, history_object

def main():
    X_train, y_train = load_training_data()
    model, history_object = build_NVIDIA(X_train, y_train)

    model.save('model.h5')

    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

    pass

if __name__ == "__main__":
    # execute only if run as a script
    main()
