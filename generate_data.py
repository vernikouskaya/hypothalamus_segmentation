
from __future__ import print_function

import os
import platform
import numpy as np

validation_folders = ['Folder1']


partition = {'train': [], 'validation': []}
data_set = {}

#save dict-file in folder data
if not os.path.exists('./metadata_training'):
    os.makedirs('./metadata_training')

#go to training data folders
cwd = os.getcwd()
os.chdir('../DATA')

system = platform.system()
dirSeparator = '/'
if system.lower() == "windows":
    dirSeparator = '\\'

i = 0
#
#get all folders
video_sets = next(os.walk('.'))[1]
for x in video_sets:
    #print(x)
    os.chdir(x)
    frame_sets = next(os.walk('.'))[1]

    for y in frame_sets:
        # relative path
        rel_path = os.getcwd().rsplit(dirSeparator, 1)
        rel_path = '../DATA/' + rel_path[-1]

        data_set['id-' + str(i)] = [rel_path + '/' + str(y) + '/mri_orig.png', rel_path + '/' + str(y) + '/mri_hypothal.png']

        print(data_set['id-' + str(i)])

        if x not in validation_folders:
            partition['train'].append('id-' + str(i))
        else:
            partition['validation'].append('id-' + str(i))
        i += 1
    os.chdir('..')


os.chdir(cwd)
print('Training samples:', len(partition['train']))
print('Validation samples:', len(partition['validation']))
np.save('./metadata_training/partition.npy', partition)
np.save('./metadata_training/data_set.npy', data_set)
