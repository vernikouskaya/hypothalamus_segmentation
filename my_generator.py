'''Skript to generate Dataset from folders
adapted code from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html'''

import numpy as np
import tensorflow.keras
import cv2
import random
from datetime import datetime
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

class DataGenerator(tensorflow.keras.utils.Sequence):
    'Save passed parameters from main function'
    def __init__(self, list_IDs, data_set, batch_size=32, dim=(512,512,1), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.data_set = data_set
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        random.seed(int(datetime.now().timestamp()))

    def __len__(self):
        'Denotes the number of batches per epoch, by default all samples are trained once per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, Y_cat = self.__data_generation(list_IDs_temp)
        return X, Y_cat

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, 512, 512, 3))
        Y = np.empty((self.batch_size, 512, 512))
        Y_cat = np.empty((self.batch_size, 512, 512, self.n_classes))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            image = np.empty((1024, 1024, 3))
            mask = np.empty((1024, 1024, 1))

            sample = self.data_set[ID]

            # read images
            image[:, :] = cv2.imread(sample[0], cv2.IMREAD_COLOR)
            mask[:, :,0] = cv2.imread(sample[1], 0)

            #very important to use interpolation, otherwise pixel adges are scaled
            img_half = cv2.resize(image[:, :], (0, 0), fx=0.5, fy=0.5, interpolation = cv2.INTER_NEAREST)
            msk_half = cv2.resize(mask[:, :,0], (0, 0), fx=0.5, fy=0.5, interpolation = cv2.INTER_NEAREST)

            # save images to array
            X[i,] = img_half
            Y[i,] = msk_half

            #print(np.unique(msk_half))
            labelencoder = LabelEncoder()
            h, w = Y[i,].shape
            train_masks_reshaped = Y[i,].reshape(-1, 1)
            values_to_fit = [0, 255]
            labelencoder.fit(values_to_fit)
            train_masks_reshaped_encoded = labelencoder.transform(train_masks_reshaped)
            train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(h, w)
            Y_cat[i,] = to_categorical(train_masks_encoded_original_shape, num_classes=self.n_classes)

        X = X.astype('float32')
        X /= 255
        Y_cat = Y_cat.astype('float32')

        return X, Y_cat
