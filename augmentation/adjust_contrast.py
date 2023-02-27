import random
import cv2

import os
import shutil
import numpy as np

os.chdir('..\\DATA_extend')
original_dir = os.getcwd()

folder = next(os.walk('.'))[1]
for f in folder:

    os.chdir(original_dir + '\\' + f)

    folders = []
    filenames = []


    # get all directories within patient folder
    dirs = [x[0] for x in os.walk('.')]
    # iterate through all folders, and attach them to folders[] if it contains 2 files
    for x in dirs:
        filenames = os.listdir(x)
        a = len(filenames)
        if "mri_hypothal.png" in filenames:
            folders.append(x[2:])
    # sort for correct sequence
    folders.sort(key=int)

    i = 0
    alphas = {0.25, 0.35, 0.4, 0.5, 0.6, 0.75} # bright
    #alphas = {0.5, 0.75, 1.25, 1.5} # demi
    #alphas = {1.5, 2.0, 2.25, 2.5, 3.0} # black
    beta = 0
    # iterate through folders
    for x in folders[0:50]:
        # read files in folders
        for root, dirs, files in os.walk('./' + x):
            for name in files:
                # read reference image
                if name == 'mri_orig.png':
                    for alpha in alphas:
                        path = os.path.join(original_dir + '\\' + f, x, name)
                        image = cv2.imread(path)
                        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
                        path_new = os.path.join(original_dir + '\\' + f, str(50+i), name)
                        cv2.imwrite(path_new, adjusted)

                        path_hypo = os.path.join(original_dir + '\\' + f, x, 'mri_hypothal.png')
                        image_hypo = cv2.imread(path_hypo)
                        path_new_hypo = os.path.join(original_dir + '\\' + f, str(50 + i), 'mri_hypothal.png')
                        cv2.imwrite(path_new_hypo, image_hypo)

                        i += 1

    os.chdir('..')
