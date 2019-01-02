# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 14:38:12 2018

@author: shen1994
"""

import numpy as np

def loadFromPts(filename):
    landmarks = np.genfromtxt(filename, skip_header=3, skip_footer=1)
    landmarks = landmarks - 1

    return landmarks

def bestFitRect(points, meanS, box=None):
    if box is None:
        box = np.array([points[:, 0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max()])
    boxCenter = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2 ])

    boxWidth = box[2] - box[0]
    boxHeight = box[3] - box[1]

    meanShapeWidth = meanS[:, 0].max() - meanS[:, 0].min()
    meanShapeHeight = meanS[:, 1].max() - meanS[:, 1].min()

    scaleWidth = boxWidth / meanShapeWidth
    scaleHeight = boxHeight / meanShapeHeight
    scale = (scaleWidth + scaleHeight) / 2

    S0 = meanS * scale

    S0Center = [(S0[:, 0].min() + S0[:, 0].max()) / 2, (S0[:, 1].min() + S0[:, 1].max()) / 2]    
    S0 += boxCenter - S0Center

    return S0

def bestFit(destination, source, returnTransform=False):
    destMean = np.mean(destination, axis=0)
    srcMean = np.mean(source, axis=0)

    srcVec = (source - srcMean).flatten()
    destVec = (destination - destMean).flatten()

    a = np.dot(srcVec, destVec) / np.linalg.norm(srcVec)**2
    b = 0
    for i in range(destination.shape[0]):
        b += srcVec[2*i] * destVec[2*i+1] - srcVec[2*i+1] * destVec[2*i] 
    b = b / np.linalg.norm(srcVec)**2
    
    T = np.array([[a, b], [-b, a]])
    srcMean = np.dot(srcMean, T)

    if returnTransform:
        return T, destMean - srcMean
    else:
        return np.dot(srcVec.reshape((-1, 2)), T) + destMean        
