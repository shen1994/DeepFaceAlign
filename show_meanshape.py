# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 17:05:17 2018

@author: shen1994
"""

import cv2
import numpy as np

if __name__ == "__main__":
    
    meanshape = np.load('data/meanFaceShape.npz')["meanShape"]
    
    cv2.namedWindow("meanshape")
    
    image = cv2.imread("test.jpg", 1)
    image = cv2.resize(image, (224, 224))
    
    while(True):
        cv2.imshow("meanshape", image)
        
        for cord in meanshape:
            x = int(cord[0] + 112.0)
            y = int(cord[1] + 112.0)
            cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
    cv2.destroyAllWindows()
