# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 11:18:22 2018

@author: shen1994
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from dan_utils import get_meanshape
from dan_utils import fit_to_rect
from dan_utils import crop_resize_rotate
from dan_utils import warp_image

if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    meanshape = get_meanshape()

    align_graph_def = tf.GraphDef()
    align_graph_def.ParseFromString(open("model/pico_FaceAlign_model.pb", "rb").read())
    align_tensors = tf.import_graph_def(align_graph_def, name="")
    align_sess = tf.Session()
    opt = align_sess.graph.get_operations()
    align_x = align_sess.graph.get_tensor_by_name("align_input:0")
    align_stage1 = align_sess.graph.get_tensor_by_name("align_stage1:0")
    align_stage2 = align_sess.graph.get_tensor_by_name("align_stage2:0")
    align_keepout1 = align_sess.graph.get_tensor_by_name("align_keepout1:0")
    align_keepout2 = align_sess.graph.get_tensor_by_name("align_keepout2:0")
    align_landmark = align_sess.graph.get_tensor_by_name("Stage2/landmark_1:0")

    o_image = cv2.imread('test1.jpg', 1)
    image = np.mean(o_image, axis=2)

    image_width = image.shape[0]
    image_height = image.shape[1]

    landmark_value = fit_to_rect(meanshape, [0, 0, image_width-1, image_height-1])
    
    image, transform = crop_resize_rotate(image, 112, landmark_value, meanshape)

    landmark = align_sess.run(align_landmark, feed_dict={align_x:[np.resize(image, (112, 112, 1))], 
                                                 align_stage1:False, align_stage2:False, 
                                                 align_keepout1:0.0, align_keepout2:0.0})[0]

    landmark = np.resize(landmark, (68, 2))

    landmark = np.dot(landmark - transform[1], np.linalg.inv(transform[0]))

    cv2.namedWindow("test")
    cv2.moveWindow("test", 100, 100)

    while(True):

        for i in range(68):
            x = int(landmark[i][0])
            y = int(landmark[i][1])

            cv2.circle(o_image, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("test", o_image)
	
        '''
        min_width = np.min(np.array([o_image.shape[1], o_image.shape[0]]))
        d_landmark = np.array([[min_width/112.0, 0], [0, min_width/112.0]])
        R, r_image = warp_image(o_image, landmark, np.dot(meanshape, d_landmark))
        R = R.tolist()
        print(np.arctan(R[0][1] / R[0][0]) / np.pi * 180.0)
        cv2.imshow("test", r_image)
        '''    
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
    cv2.destroyAllWindows()
    align_sess.close()
   