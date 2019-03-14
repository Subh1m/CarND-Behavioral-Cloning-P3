import cv2
import csv
import numpy as np
import os
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D, Dropout
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

def get_lines(path, skipHeader = False):
    """
    Returns line information from driving logs
    """
    lines = []
    with open(path + '/driving_log.csv') as csvFile:
        reader = csv.reader(csvFile)
        if skipHeader:
            next(reader, None)
        for line in reader:
            lines.append(line)
    return lines

def generator(samples, batch_size = 32):
    """
    Training data generation.
    """
    num_samples = len(samples)
    while True: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for imagePath, measurement in batch_samples:
                originalImage = cv2.imread(imagePath)
                image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
                images.append(image)
                angles.append(measurement)
                # Flipping Images
                images.append(cv2.flip(image,1))
                angles.append(measurement*-1.0)

            # trim image to only see section with road
            inputs = np.array(images)
            outputs = np.array(angles)
            yield sklearn.utils.shuffle(inputs, outputs)

# Loading Directory Address
path = 'data'
folder_link = [x[0] for x in os.walk(path)]
data_folder = list(filter(lambda directory: os.path.isfile(directory + '/driving_log.csv'), folder_link))
centre_images = []
left_images = []
right_images = []
measurement_images = []
for directory in data_folder:
    lines = get_lines(directory)
    center = []
    left = []
    right = []
    measurements = []
    # Loading images (center, left, right)
    for line in lines[1:]:
        measurements.append(float(line[3]))
        center.append(directory + '/' + line[0].strip())
        left.append(directory + '/' + line[1].strip())
        right.append(directory + '/' + line[2].strip())
    centre_images.extend(center)
    left_images.extend(left)
    right_images.extend(right)
    measurement_images.extend(measurements)
    
# Correction for the distance in left and right images w.r.t. center image
correction = 0.2
image_path = []
image_path.extend(centre_images)
image_path.extend(left_images)
image_path.extend(right_images)
measurements = []
measurements.extend(measurement_images)
measurements.extend([x + correction for x in measurement_images])
measurements.extend([x - correction for x in measurement_images])

samples = list(zip(image_path, measurements))
train_samples, validation_samples = train_test_split(samples, test_size = 0.2)

# Data Generation without overloading memory
train_generator = generator(train_samples, batch_size = 128)
validation_generator = generator(validation_samples, batch_size = 128)

# Model Architecture
model = Sequential()
model.add(Lambda(lambda x: (x/127.5)-1.0, input_shape = (160, 320, 3)))
model.add(Cropping2D(cropping=((50, 20), (0, 0))))
model.add(Conv2D(24, 5, 5, subsample = (2, 2), activation = 'relu'))
model.add(Conv2D(36, 5, 5, subsample = (2, 2), activation = 'relu'))
model.add(Conv2D(48, 5, 5, subsample = (2, 2), activation = 'relu'))
model.add(Conv2D(64, 3, 3, activation = 'relu'))
model.add(Conv2D(64, 3, 3, activation = 'relu'))
# Reducing overfitting
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss = 'mse', optimizer = 'adam')
print(model.summary())

# Callbacks fro Best Model using ModelCheckpoint
checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# Using fit_generator to train instead of fit, storing model.h5 using ModelCheckpoint
model.fit_generator(train_generator, samples_per_epoch = len(train_samples), callbacks = [checkpoint], validation_data = validation_generator, nb_val_samples = len(validation_samples), nb_epoch = 5, verbose = 1)

# Model storage of the final version (not best model)
model.save('model_diff.h5')