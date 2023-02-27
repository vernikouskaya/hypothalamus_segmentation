import os
import numpy as np
import cv2
from operator import itemgetter
from sklearn.preprocessing import LabelEncoder

folder = next(os.walk('.'))[1]
for f in folder:

    # Declarations
    X = []
    Y = []
    folders = []
    filenames = []
    readMask = False
    readImage = False
    image = np.empty((1024, 1024, 3))
    mask = np.empty((1024, 1024, 1))

    cwd = os.getcwd()

    # Jump into folder
    os.chdir(f)
    # get all directories within patient folder
    dirs = [x[0] for x in os.walk('.')]
    # iterate through all folders, and attach them to folders[] if it contains 2 files
    for x in dirs:
        filenames = os.listdir(x)
        a = len(filenames)
        if "mri_orig.png" in filenames and "mri_hypothal.png" in filenames:
            folders.append(x[2:])
    # sort for correct sequence
    folders.sort(key=int)


    # iterate through folders
    for x in folders:
        # read files in folders
        for root, dirs, files in os.walk('./' + x):
            for name in files:
                # read reference image
                if name == 'mri_hypothal.png':
                    mask[:, :, 0] = cv2.imread(root + '/' + name, 0)
                    readMask = True
                    mask_half = cv2.resize(mask[:, :, 0], (0, 0), fx=0.5, fy=0.5, interpolation = cv2.INTER_NEAREST)

                    labelencoder = LabelEncoder()
                    h, w = mask_half.shape
                    train_masks_reshaped = mask_half.reshape(-1, 1)
                    values_to_fit = [0, 255]
                    labelencoder.fit(values_to_fit)
                    train_masks_reshaped_encoded = labelencoder.transform(train_masks_reshaped)
                    train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(h, w)

                    Y.append(train_masks_encoded_original_shape)
                    mask = np.empty((1024, 1024, 1))

                if name == 'mri_orig.png':
                    image[:, :] = cv2.imread(root + '/' + name, cv2.IMREAD_COLOR)
                    readImage = True
                    image_half = cv2.resize(image[:, :], (0, 0), fx=0.5, fy=0.5, interpolation = cv2.INTER_NEAREST)
                    X.append(image_half)
                    image = np.empty((1024, 1024, 3))
                if readMask and readImage:
                    readMask = False
                    readImage = False
                    print('Y shape', np.array(Y).shape)
                    print('X shape:', np.array(X).shape)

    os.chdir('..')

    #save as npz file
    np.savez_compressed(f, x_test=X, y_test=Y)

    os.chdir(cwd)
