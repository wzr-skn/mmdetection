import scipy.io as scio
import sys
import os
import json
import numpy as np
datafile = '/media/traindata/hands_datasets/oxford-hand-dataset/hand_dataset/hand_dataset/test_dataset/test_data/annotations/voc2007_95.mat'
data = scio.loadmat(datafile)
print(data)