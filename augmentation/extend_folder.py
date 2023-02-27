import os
import numpy as np
import cv2

os.chdir('..\\DATA_extend')
target_dir = os.getcwd()

folder = next(os.walk('.'))[1]
for x in folder:
    #for ext in range(50,250): # medi
    for ext in range(50, 350):  # bright
    #for ext in range(50, 300):  # black
        directory = str(ext)
        path = os.path.join(target_dir, x, directory)
        if not os.path.exists(path):
            os.makedirs(path)
            os.chdir(target_dir)

