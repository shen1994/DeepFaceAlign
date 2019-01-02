# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 13:10:28 2018

@author: shen1994
"""

import cv2
import pickle
import numpy as np
from up_utils import image_landmark_generate
from up_utils import landmark_to_gtlandmark
from up_utils import perturbation_generate
# from up_utils import image_normalize

if __name__ == "__main__":


    with open('data/train_filenames.pkl', 'rb') as f:
        filenames = pickle.load(f)
    with open('data/train_landmarks.pkl', 'rb') as f:
        landmarks = pickle.load(f)
        
    meanshape = np.load('data/meanFaceShape.npz')["meanShape"]
    meanshape = (meanshape + 112) / 2.0

    image, landmark = image_landmark_generate(filenames[2], landmarks[2], is_fliplr=True)
    gtlandmark = landmark_to_gtlandmark(landmark, meanshape)
    
    image, landmark, gtlandmark = perturbation_generate(image, landmark, gtlandmark, meanshape, [0.2, 0.2, 20, 0.25])

    # image = image_normalize(image)

    cv2.namedWindow("test")
    
    while(True):
            
        for cord in landmark:
            x = int(cord[0])
            y = int(cord[1])

            cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
        cv2.imshow("test", image)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
    cv2.destroyAllWindows() 