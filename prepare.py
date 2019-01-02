# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 11:06:12 2018

@author: shen1994
"""

import os
import glob
import pickle
from utils import loadFromPts

def build_dataset(path_list, box_path_list, is_train=True):
    
    filenames = []
    landmarks = []
    faceboxes = []
    for i in range(len(path_list)):
        filenames_dir = glob.glob(path_list[i] + "*.jpg")
        filenames_dir += glob.glob(path_list[i] + "*.png")
        
        box_dict = pickle.load(open(box_path_list[i], 'rb'))

        for j in range(len(filenames_dir)):
            filenames.append(filenames_dir[j])
            pts_filename = filenames_dir[j][:-3] + "pts"
            landmarks.append(loadFromPts(pts_filename))
            basename = os.path.basename(filenames_dir[j])
            faceboxes.append(box_dict[basename])
    
    if is_train:
        with open('data/train_filenames.pkl', 'wb') as f:
            pickle.dump(filenames, f, pickle.HIGHEST_PROTOCOL)
        with open('data/train_faceboxes.pkl', 'wb') as f:
            pickle.dump(faceboxes, f, pickle.HIGHEST_PROTOCOL)
        with open('data/train_landmarks.pkl', 'wb') as f:
            pickle.dump(landmarks, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open('data/valid_filenames.pkl', 'wb') as f:
            pickle.dump(filenames, f, pickle.HIGHEST_PROTOCOL)
        with open('data/valid_faceboxes.pkl', 'wb') as f:
            pickle.dump(faceboxes, f, pickle.HIGHEST_PROTOCOL)
        with open('data/valid_landmarks.pkl', 'wb') as f:
            pickle.dump(landmarks, f, pickle.HIGHEST_PROTOCOL)
    
if __name__ == "__main__":

    trainset_path = ['data/afw/', 'data/ibug/', \
                    'data/300W/01_Indoor/', 'data/300W/02_Outdoor/', \
                    'data/lfpw/trainset/', 'data/helen/trainset/']
    trainset_box_path = ['data/py3boxesAFW.pkl', 'data/py3boxesIBUG.pkl', \
                        'data/py3boxes300WIndoor.pkl', 'data/py3boxes300WOutdoor.pkl', \
                        'data/py3boxesLFPWTrain.pkl', 'data/py3boxesHelenTrain.pkl']
    build_dataset(trainset_path, trainset_box_path, is_train=True)
    
    validset_path = ['data/lfpw/testset/', 'data/helen/testset/']
    validset_box_path = ['data/py3boxesLFPWTest.pkl', 'data/py3boxesHelenTest.pkl']
    build_dataset(validset_path, validset_box_path, is_train=False)
