#filename : test_local_performance.py
#author : PRAJWAL T R
#date last modified : Sun Nov 15 19:59:35 2020
#comments :

'''
    analyse accuracy paradox due to uneven classes in train dataset
'''

# imports
from os import walk
from local_model import getLocalModel
import numpy as np

# constants
nextxy_classes = 25 # 25 classes ex : 0 to 24 in 5 * 5 grid
touch_classes = 2 # two classes ex : 1, 0
local_gen_path = "../kanjivg_dataset/"
kanjivg_samples_path = "../kanjivg_dataset/kanji_modified/"
touch_thresh = 0.5

# add path to local stroke generator
from sys import path
path.append(local_gen_path)

# import local stroke generator
from local_strokegenerator import strokeGenerator
from test_local_model import prepInput

# prep nessescary structures
nextxy_tp = {}
touch_tp = {}
nextxy_count = {}
touch_count = {}

for nxyclass in range(nextxy_classes):
    nextxy_tp[nxyclass] = 0 # true positives ex: correct predictions
    nextxy_count[nxyclass] = 0 # all samples with for that class

for tclass in range(touch_classes):
    touch_tp[tclass] = 0 # true positives ex: correct predictions
    touch_count[tclass] = 0 # all samples with for that class

# get local model with trained weights loaded
local_model = getLocalModel()

# get generator that generates samples for local model
_, _, filelist = next(walk(kanjivg_samples_path))
filelist = filelist[::-1][:20] # get 200 samples from last
sg = strokeGenerator(filelist)

while True: # till samples exhaust
    try:
        inp, ext, touch, next_xy = next(sg)
        touch_pred, next_xy_pred = local_model.predict([np.expand_dims(inp, axis=0), np.expand_dims(ext, axis=0)])
        if np.argmax(next_xy) == np.argmax(next_xy_pred[0]): # true positive
            nextxy_tp[np.argmax(next_xy)] += 1
        nextxy_count[np.argmax(next_xy)] += 1 # total count for that sample

        if touch_pred[0] > touch_thresh and touch == 1:
            touch_tp[1] += 1 # true postive for class 1
        if touch_pred[0] <= touch_thresh and touch == 0:
            touch_tp[0] += 1 # true postive for class 0
        touch_count[touch[0]] += 1

    except StopIteration:
        break
# print recall stats for each class
print("\nrecall for each class in next xy grid : \n")

for nxyclass in range(nextxy_classes):
    print("class : ", nxyclass, ", recall : ", nextxy_tp[nxyclass]/nextxy_count[nxyclass] if nextxy_count[nxyclass] != 0 else 0)

# print recall stats for each class in touch
print("\nrecall for each touch class : \n")

for touch in range(touch_classes):
    print("class : ", touch, ", recall : ", touch_tp[touch]/touch_count[touch])
