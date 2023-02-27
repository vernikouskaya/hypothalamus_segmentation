'''Trains a simple convnet on a custom dataset'''


from __future__ import print_function

from datetime import datetime
import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Flatten, Input, BatchNormalization,Activation
from keras.layers import Conv2D, MaxPooling2D, Conv3D, UpSampling2D, Concatenate, Conv2DTranspose, Add
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

from keras.utils.vis_utils import plot_model

from my_generator import DataGenerator
import os
import random
from random import choices
import cv2
#import model_test

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
from skimage.color import rgb2gray
from tensorflow.python.keras.callbacks import ModelCheckpoint


import segmentation_models as sm

# Parameters
params = {'dim': (1024, 1024, 3),
          'batch_size': 4,
          'n_classes': 2,
          'n_channels': 1,
          'shuffle': True}
plt.cla()

# Load Datasets
print(os.getcwd())
partition = np.load('./metadata_training/partition.npy', allow_pickle=True).item()
data_set = np.load('./metadata_training/data_set.npy', allow_pickle=True).item()

val = partition['train']
image = np.empty((1024,1024,1))
mask = np.empty((1024,1024,1))

X_test = []

number_of_plots = 3
plt.rcParams['figure.figsize'] = [14, 10]
f, axs = plt.subplots(number_of_plots, number_of_plots)
i = 0
k = 1
for sample in val:
    sample = data_set[sample]

    if k % 2000 == 0:
        image[:, :, 0] = cv2.imread(sample[0], 0)
        mask[:, :, 0] = cv2.imread(sample[1], 0)


        if i < (number_of_plots * number_of_plots):

            axs[i // number_of_plots, i % number_of_plots].imshow(image[:, :, 0], cmap=plt.cm.gray,
                                                                  interpolation='none')
            axs[i // number_of_plots, i % number_of_plots].imshow(mask[:, :, 0], cmap=plt.cm.jet, interpolation='none',
                                                                  alpha=0.3)
            axs[i // number_of_plots, i % number_of_plots].axis('off')
        i += 1
        k += 1
    else:
        k += 1

    X_test.append(image)

plt.show()
f.savefig("Training samples.png", type="png", dpi=300)
print(len(val))
print('Number Samples: ', len(X_test))


# Instantiate Generators
training_generator = DataGenerator(partition['train'], data_set, **params)
validation_generator = DataGenerator(partition['validation'], data_set, **params)

BACKBONE = 'efficientnetb0'
preprocess_input = sm.get_preprocessing(BACKBONE)
model = sm.Unet(BACKBONE, encoder_weights='imagenet', classes=2, activation='softmax')
model.summary()

model.compile(
    'Adam',
    loss=sm.losses.cce_jaccard_loss,
    metrics=[sm.metrics.iou_score],
)

### checkpoint
filepath="efficientnetb0_{epoch:02d}-{val_loss:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1)

training_time_start = datetime.now()
print("Training started at: {}".format(training_time_start))
# Train model
history = model.fit(x=training_generator,
                              validation_data=validation_generator,
                              use_multiprocessing=False,
                              workers=1, epochs=25, callbacks=[checkpoint, early_stop],
                              verbose=1)

training_time_stop = datetime.now()
print("Training stopped at: {}".format(training_time_stop))
print("Total training time: {}".format(training_time_stop - training_time_start))

timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
# Save model

model.save('efficientnetb0.h5')

print(history.history.keys())

# Plot the LOSS and IoU curves
plt.figure(0)
plt.plot(history.history['loss'], 'bo--')
plt.plot(history.history['val_loss'], 'ro-')
plt.ylabel('LOSS [cce_jaccard_loss]')
plt.xlabel('Epochs (n)')
plt.legend(['Training', 'Validation'])
plt.savefig('Loss.png', dpi=150)

plt.figure(1)
plt.plot(history.history['iou_score'], 'bo--')
plt.plot(history.history['val_iou_score'], 'ro-')
plt.ylabel('IoU Score')
plt.xlabel('Epochs (n)')
plt.legend(['Training', 'Validation'])
plt.savefig('IoU.png', dpi=150)

#plt.close("all")