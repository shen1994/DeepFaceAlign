# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 13:02:53 2018

@author: shen1994
"""

import pickle
import tensorflow as tf
import numpy as np

from up_utils import image_landmark_generate
from up_utils import landmark_to_gtlandmark
from up_utils import perturbation_generate

def create_tfrecord(filenames_path, landmarks_path, faceboxes_path, out_path):
    
    writer = tf.python_io.TFRecordWriter(out_path)
    
    with open(filenames_path, 'rb') as f:
        filenames = pickle.load(f)
    with open(landmarks_path, 'rb') as f:
        landmarks = pickle.load(f)
    #with open(faceboxes_path, 'rb') as f:
    #    faceboxes = pickle.load(f)

    meanshape = np.load('data/meanFaceShape.npz')["meanShape"]

    for i in range(len(filenames)):
        # not flip and generate 10 images
        o_image, o_landmark = image_landmark_generate(filenames[i], landmarks[i])
        o_gtlandmark = landmark_to_gtlandmark(o_landmark, meanshape) 
        for j in range(10):
            image, landmark, gtlandmark = perturbation_generate(o_image, o_landmark, o_gtlandmark, meanshape, [0.2, 0.2, 20, 0.25])
            image = np.array(image)
            landmark = np.array(landmark)
            landmark = np.resize(landmark, (landmark.shape[0] * landmark.shape[1]))

            img_raw = image.tobytes()  
            example = tf.train.Example(features = tf.train.Features(feature = {
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                    'landmark': tf.train.Feature(float_list=tf.train.FloatList(value=landmark))
                    }))
            writer.write(example.SerializeToString())
         
        # flip and generate 10 images
        o_image, o_landmark = image_landmark_generate(filenames[i], landmarks[i], is_fliplr=True)
        o_gtlandmark = landmark_to_gtlandmark(o_landmark, meanshape) 
        for j in range(10):
            image, landmark, gtlandmark = perturbation_generate(o_image, o_landmark, o_gtlandmark, meanshape, [0.2, 0.2, 20, 0.25])
            image = np.array(image)
            landmark = np.array(landmark)
            landmark = np.resize(landmark, (landmark.shape[0] * landmark.shape[1]))

            img_raw = image.tobytes()  
            example = tf.train.Example(features = tf.train.Features(feature = {
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                    'landmark': tf.train.Feature(float_list=tf.train.FloatList(value=landmark))
                    }))
            writer.write(example.SerializeToString())

        print(u'dataset:' + str(i) + u'---OK!')

    writer.close()

if __name__ == "__main__":
    
    create_tfrecord("data/train_filenames.pkl", \
                    "data/train_landmarks.pkl", \
                    "data/train_faceboxes.pkl", \
                    "data/train_dataset.tfrecords")
    create_tfrecord("data/valid_filenames.pkl", \
                    "data/valid_landmarks.pkl", \
                    "data/valid_faceboxes.pkl", \
                    "data/valid_dataset.tfrecords")