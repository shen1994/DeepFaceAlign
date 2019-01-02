# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 16:55:45 2018

@author: shen1994
"""

import cv2
import numpy as np
from scipy import ndimage
from utils import mirrorShape
from utils import bestFit
from utils import bestFitRect

def landmark_to_gtlandmark(landmark, meanshape, box=None, mode='rect'):
    
    best_fit = []
    
    if mode == 'rect':
        best_fit = bestFitRect(landmark, meanshape)
    elif mode == 'similarity':
        best_fit = bestFit(landmark, meanshape)
    elif mode == 'box':
        best_fit = bestFitRect(landmark, meanshape, box)

    return best_fit
    
def image_landmark_generate(filename, landmark, is_fliplr=False):
    
    image = cv2.imread(filename, 1)
    image = np.mean(image, axis=2)
    image = image.astype(np.uint8)
    if is_fliplr:
        image = np.fliplr(image)
        landmark = mirrorShape(landmark, image.shape)
    return image, landmark
    
def crop_resize_rotate(image, image_size, landmark, inputshape):
    
    A, t = bestFit(inputshape, landmark, True)
    A2 = np.linalg.inv(A)
    t2 = np.dot(-t, A2)

    out_image = ndimage.interpolation.affine_transform(image, \
                                                       A2, t2[[1, 0]], \
                                                       output_shape=[image_size, image_size])
    return out_image, [A, t]
    
def perturbation_generate(image, landmark, gtlandmark, meanshape, perturbation, image_size=112, fraction=0.25):
    '''
    do random transformation for image and landmark
    '''
    meanshape_size = max(meanshape.max(axis=0) - meanshape.min(axis=0))
    dest_shapesize = image_size * (1 - 2 * fraction)
    scale_shapesize = meanshape * (dest_shapesize / meanshape_size)
    
    move_x = perturbation[0] * (scale_shapesize[:, 0].max() - scale_shapesize[:, 0].min())
    move_y = perturbation[1] * (scale_shapesize[:, 1].max() - scale_shapesize[:, 1].min())
    offset = [np.random.normal(0, move_x), np.random.normal(0, move_y)]
    
    rotation_pad = perturbation[2] * np.pi / 180.0
    angle = np.random.normal(0, rotation_pad)
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    
    scale = np.random.normal(1, perturbation[3])
    
    gtlandmark = gtlandmark + offset
    gtlandmark = (gtlandmark - gtlandmark.mean(axis=0)) * scale + gtlandmark.mean(axis=0)            
    gtlandmark = np.dot(R, (gtlandmark - gtlandmark.mean(axis=0)).T).T + gtlandmark.mean(axis=0)
    
    dest_shape = scale_shapesize - scale_shapesize.mean(axis=0)
    dest_shape = dest_shape + np.array([image_size, image_size]) / 2.0

    out_image, [A, t] = crop_resize_rotate(image, image_size, gtlandmark, dest_shape)
 
    landmark = np.dot(landmark, A) + t

    gtlandmark = np.dot(gtlandmark, A) + t

    return out_image, landmark, gtlandmark
    
def image_normalize(image):
    
    image = image.astype(np.float32)
    image = (image - np.mean(image, axis=0)) / np.std(image, axis=0)
    
    return image
    