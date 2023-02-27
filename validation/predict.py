'''Trains a simple convnet on a custom dataset'''

from __future__ import print_function

import tensorflow
from keras.models import load_model

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import os
import cv2
from numpy.linalg import inv
from skimage import *
from skimage import color
import scipy
from scipy.spatial.distance import directed_hausdorff
# import pip
# pip.main(['install', 'imutils'])
from scipy.spatial import distance
import time
import segmentation_models as sm
from keras.metrics import MeanIoU
import random
import pandas as pd
import tensorflow as tf
import surface_distance

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import seaborn as sns

print(tensorflow.__version__)
tensorflow.config.list_physical_devices('GPU')


input_shape = (512, 512, 3)

# function to load npz files
def load_data_validation(path):
    f = np.load(path)
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_test, y_test)


model = load_model('efficientnetb0.h5', compile=False)


cwd = os.getcwd()
# get all files in directory
files = [f for f in os.listdir('.') if os.path.isfile(f)]

x_test_half = []
y_test_half = []

df = pd.DataFrame([])

for f in files:
    # iterate through all npz files
    if '.npz' in f:
        # print processed filename
        print(f)
        folders = []

        folder_path = cwd + "\\" + f.rsplit('.npz', -1)[0]
        folders = os.listdir(folder_path)
        # sort for correct sequence
        folders.sort(key=int)

        # load testfile
        (x_test, y_test) = load_data_validation(cwd + "\\" + f)
        x_test = x_test.astype('float32')
        x_test /= 255

        prediction_time_start = time.process_time()
        # predict output
        y_pred = model.predict(x_test)
        #y_pred1 = model.predict(x_test[0:25])
        #y_pred2 = model.predict(x_test[25:50])
        #y_pred = np.concatenate((y_pred1,y_pred2),axis=0)
        prediction_time_stop = time.process_time()
        print("Prediction time per frame [s]: {}".format(
            (prediction_time_stop - prediction_time_start) / x_test.shape[0]))
        gpu_dict = tf.config.experimental.get_memory_info('GPU:0')
        current = float(gpu_dict['current']) / (1024 ** 3)
        peak = float(gpu_dict['peak']) / (1024 ** 3)
        y_predict = np.argmax(y_pred, axis=3)

        pixel_count = np.empty(len(y_predict))
        pixel_count_GT = np.empty(len(y_predict))
        count = []
        GT = []

        i = 0
        k = 1
        number_of_plots = 2
        plt.rcParams['figure.figsize'] = [14, 10]
        fig, axs = plt.subplots(number_of_plots, number_of_plots)


        for fr in range(len(y_predict)):

            os.chdir(folder_path + '\\' + str(folders[fr]))

            norm_image = cv2.normalize(y_predict[fr, :, :], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                       dtype=cv2.CV_32F)
            im_prediction_scaled = norm_image.astype(np.uint8)
            cv2.imwrite('predicted.png', im_prediction_scaled)

            pixel_count[fr] = np.count_nonzero(y_predict[fr, :, :], axis=None)
            pixel_count_GT[fr] = np.count_nonzero(y_test[fr, :, :], axis=None)

            if (pixel_count[fr] != 0):
                count.append(pixel_count[fr])

            if (pixel_count_GT[fr] != 0):
                GT.append(pixel_count_GT[fr])



            im_prediction = y_predict[fr, :, :]
            if k % 5 == 0:
                if i < (number_of_plots * number_of_plots):
                    axs[i // number_of_plots, i % number_of_plots].imshow(im_prediction, cmap=plt.cm.gray,
                                                                          interpolation='none')
                    axs[i // number_of_plots, i % number_of_plots].axis('off')
                    axs[i // number_of_plots, i % number_of_plots].set_title(str(folders[fr + 1]))
                i += 1
                k += 1
            else:
                k += 1

        plt.show()

        os.chdir(cwd)
        # To calculate I0U for each class...
        n_classes = 2
        IOU_keras = MeanIoU(num_classes=n_classes)
        IOU_keras.update_state(y_test[:, :, :], y_predict[:, :, :])
        print("Average IOU = ", IOU_keras.result().numpy())

        surface_distances = surface_distance.compute_surface_distances(
            y_test[:, :, :].astype(bool), y_predict[:, :, :].astype(bool), spacing_mm=(0.125, 0.125, 0.5))

        hausdorff = surface_distance.compute_robust_hausdorff(
            surface_distances, 95)
        print("95%HD is: ", hausdorff)
        dice = surface_distance.compute_dice_coefficient(
            y_test[:, :, :], y_predict[:, :, :])
        print("dice is: ", dice)

        FP = len(np.where(y_predict - y_test == 1)[0]) #FP: ground truth pixel=0 while predicted pixel=1 false positive (m01)
        FN = len(np.where(y_predict - y_test == -1)[0]) #FN: ground truth pixel=1 while predicted pixel=0 false negative (m10)
        TP = len(np.where(y_predict + y_test == 2)[0]) # TP: ground truth and predicted pixel are of class 1 (object) true positive (m11)
        TN = len(np.where(y_predict + y_test == 0)[0]) #TN: ground truth and predicted pixel are of class 0 (background) true negative (m00)
        cmat = [[TN, FP], [FN, TP]]

        Precision = TP / (TP + FP)
        Recall = TP/(TP+FN)
        Accuracy = (TP+TN)/(TP+TN+FP+FN)
        f1_score = 2*TP/ (2*TP+FP+FN)
        print('Precision: ', Precision) #purity of positive detections relative to the ground truth: what fraction of predictions as a positive class were actually positive
        print('Recall: ', Recall) # completeness of positive predictions relative to the ground truth: what fraction of all positive samples were correctly predicted as positive by the classifier
        print('Accuracy', Accuracy) #percent of pixels in the image which were correctly classified
        print('F1-score', f1_score)  # harmonic mean of precision and recall DICE
        plt.figure(figsize=(6, 6))
        sns.heatmap(cmat / np.sum(cmat), cmap="Reds", annot=True, fmt='.2%', square=1, linewidth=2.)
        plt.xlabel("predictions")
        plt.ylabel("ground truth")
        #plt.show()
        #plt.savefig(f[:-4] + '_confusion_matrix.png', dpi=300)

        size_first_half = len(count) // 2
        size_first_half_GT = len(GT) // 2

        sm_count_first = sum(count[0:int(size_first_half)])
        sm_count_GT_first = sum(GT[0:int(size_first_half_GT)])
        sm_count_second = sum(count[int(size_first_half):(len(count))])
        sm_count_GT_second = sum(GT[int(size_first_half_GT):(len(GT))])


        print('n_pixel prediction:', np.sum(pixel_count))
        print('n_pixel GT:', np.sum(pixel_count_GT))
        print('n_pixel_first prediction:', sm_count_first)
        print('n_pixel first GT:', sm_count_GT_first)
        print('n_pixel_second prediction:', sm_count_second)
        print('n_pixel second GT:', sm_count_GT_second)

        test_img_number = random.randint(0, len(x_test))
        test_img_number = 15
        test_img = x_test[test_img_number]
        ground_truth = y_test[test_img_number]
        pred = y_predict[test_img_number]

        fig1, axs1 = plt.subplots(1, 3)
        plt.subplot(131)
        plt.title('Test Image')
        plt.imshow(test_img[:, :, 0], cmap='gray')
        plt.axis('off')
        plt.subplot(132)
        plt.title('Ground Truth')
        plt.imshow(ground_truth, cmap='jet')
        plt.axis('off')
        plt.subplot(133)
        plt.title('Prediction EfficientNetB0')
        plt.imshow(pred, cmap='jet')
        plt.axis('off')
        plt.show()
        fig1.savefig(f[:-4] + '_Test for folder ' + str(test_img_number+1) + '.png', dpi=300)

        df = df.append(
            pd.DataFrame({'prediction time': (prediction_time_stop - prediction_time_start) / x_test.shape[0],
                          'peak memory [GB]': peak,
                          'average IoU': IOU_keras.result().numpy(), 'volumetric dice': dice, 'hausdorff D': hausdorff,
                          'pxl count prediction': np.sum(pixel_count), 'pxl count GT': np.sum(pixel_count_GT),
                          'TP': TP, 'FN': FN, 'FP': FP, 'TN': TN,'Precision': Precision, 'Recall': Recall, 'Accuracy': Accuracy, 'F1-score': f1_score,
                          'pxl count 1st half prediction': sm_count_first,
                          'pxl count 1st half GT': sm_count_GT_first, 'pxl count 2nd half prediction': sm_count_second,
                          'pxl count 2nd half GT': sm_count_GT_second}, index=[f]))
print(df)
df.to_csv('RESULTS.csv', index=True)