# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 14:38:12 2018

@author: shen1994
"""

import os
import pickle
import numpy as np
import tensorflow as tf

from models import DAN
# from up_utils import image_normalize

def init_meanshape():

    meanshape = np.array(
        [18.494846, 36.61314,  18.742294, 46.48093,  19.842281, 56.306053, 21.901043, 65.96931 , 
         25.723728, 74.954796, 31.63908 , 82.74449,  38.918457, 89.24516 , 46.927822, 94.50676 ,
         56.      , 96.05594 , 65.072174, 94.50676,  73.08154,  89.24516 , 80.360916, 82.74449 ,
         86.276276, 74.954796, 90.09895 , 65.96931,  92.157715, 56.306053, 93.257706, 46.48093 ,
         93.50516 , 36.61314 , 25.466898, 29.288471, 30.161482, 25.041758, 36.789574, 23.795765,
         43.62364 , 24.792725, 50.020443, 27.470646, 61.979557, 27.470646, 68.37636 , 24.792725,
         75.21043,  23.795765, 81.83852 , 25.041758, 86.533104, 29.288471, 56.      , 35.240334,
         56.      , 41.667118, 56.      , 48.046043, 56.      , 54.62276 , 48.454185, 58.960278,
         52.086994, 60.277733, 56.      , 61.446247, 59.913006, 60.277733, 63.545815, 58.960278,
         33.0997  , 35.995438, 37.131203, 33.620655, 42.01633,  33.695942, 46.26822 , 36.990746,
         41.678383, 37.84901 , 36.82142 , 37.774963, 65.73178 , 36.990746, 69.98367 , 33.695942,
         74.8688  , 33.620655, 78.9003,   35.995438, 75.17858 , 37.774963, 70.32162 , 37.84901 , 
         41.455666, 70.87096,  46.8126  , 68.76418 , 52.228645, 67.59149 , 56.      , 68.56457 , 
         59.771355, 67.59149 , 65.1874  , 68.76418 , 70.544334, 70.87096 , 65.35278 , 76.01848 ,
         60.10567 , 78.26798 , 56.      , 78.70176 , 51.89433 , 78.26798 , 46.647213, 76.01848 , 
         43.710133, 71.16683 , 52.17189 , 70.80757 , 56.      , 71.22449,  59.82811 , 70.80757 , 
         68.28986 , 71.16683 , 59.899345, 73.41396 , 56.,       73.87833 , 52.100655, 73.41396])
    
    return meanshape
    
def samples_counter():
    
    with open('data/train_filenames.pkl', 'rb') as f:
        filenames = pickle.load(f)
    train_samples_number = len(filenames) * 2 * 10

    with open('data/valid_filenames.pkl', 'rb') as f:
        filenames = pickle.load(f)
    valid_samples_number = len(filenames) * 2 * 10
    
    return train_samples_number, valid_samples_number
    
def read_and_decode(file_name, batch_size):
    
    filename_queue = tf.train.string_input_producer([file_name], shuffle=True, num_epochs = None)
    reader = tf.TFRecordReader()  
    _, serialized_example = reader.read(filename_queue)   
    
    features = tf.parse_single_example(serialized_example, features={ 
                'image': tf.FixedLenFeature([], tf.string),
                'landmark': tf.FixedLenFeature([136], tf.float32),
                })
    image = tf.decode_raw(features['image'], tf.uint8)
    image = tf.reshape(image,[112, 112, 1])
    image = tf.cast(image, tf.float32)
    
    landmark = tf.cast(features['landmark'], tf.float32)
    
    image_batch, landmark_batch = tf.train.shuffle_batch([image, landmark], 
                                                      batch_size = batch_size,
                                                      num_threads = 8,
                                                      capacity = 500 + 3 * batch_size,
                                                      min_after_dequeue = 500)
    return image_batch, landmark_batch

if __name__ == "__main__":
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    
    log_path = 'logs'
    if not os.path.exists(log_path):
        os.mkdir(log_path)
        
    model_path = 'model'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
        
    train_number, valid_number = samples_counter()
    
    nb_epocs = 20
    batch_size = 128
    stage = 2

    meanshape = init_meanshape()

    model = DAN(meanshape)

    train_image, train_landmark = read_and_decode('data/train_dataset.tfrecords', batch_size)
    valid_image, valid_landmark = read_and_decode('data/valid_dataset.tfrecords', batch_size)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()  
        threads = tf.train.start_queue_runners(sess = sess, coord = coord)

        if stage >= 2:
            saver.restore(sess,'model/model_step2_02.ckpt')
        
        for one_epoc in range(nb_epocs):
            for step in range(int(train_number / batch_size)):
                t_image, t_landmark = sess.run([train_image, train_landmark])
                # t_image = image_normalize(t_image)
                if stage < 2:              
                    sess.run(model['S1_Optimizer'], \
                             feed_dict={model['InputImage']:t_image, model['GroundTruth']:t_landmark, \
                                        model['S1_Keepout']:0.5, model['S2_Keepout']:0.5, \
                                        model['S1_isTrain']:True, model['S2_isTrain']:False})
                else:
                    sess.run(model['S2_Optimizer'], \
                             feed_dict={model['InputImage']:t_image, model['GroundTruth']:t_landmark, \
                                        model['S1_Keepout']:0.5, model['S2_Keepout']:0.5, \
                                        model['S1_isTrain']:False, model['S2_isTrain']:True})
                if step % 100 == 0:
                    v_image, v_landmark = sess.run([valid_image, valid_landmark])
                    # v_image = image_normalize(v_image)
                    if stage < 2:
                        t_cost = sess.run(model['S1_Cost'], \
                                         feed_dict={model['InputImage']:t_image, model['GroundTruth']:t_landmark, \
                                                    model['S1_Keepout']:0.0, model['S2_Keepout']:0.0, \
                                                    model['S1_isTrain']:False, model['S2_isTrain']:False})
                        v_cost = sess.run(model['S1_Cost'], \
                                         feed_dict={model['InputImage']:v_image, model['GroundTruth']:v_landmark, \
                                                    model['S1_Keepout']:0.0, model['S2_Keepout']:0.0, \
                                                    model['S1_isTrain']:False, model['S2_isTrain']:False})
                    else:
                        t_cost = sess.run(model['S2_Cost'], \
                                         feed_dict={model['InputImage']:t_image, model['GroundTruth']:t_landmark, \
                                                    model['S1_Keepout']:0.0, model['S2_Keepout']:0.0, \
                                                    model['S1_isTrain']:False, model['S2_isTrain']:False})                        
                        v_cost = sess.run(model['S2_Cost'], \
                                 feed_dict={model['InputImage']:v_image, model['GroundTruth']:v_landmark, \
                                            model['S1_Keepout']:0.0, model['S2_Keepout']:0.0, \
                                            model['S1_isTrain']:False, model['S2_isTrain']:False})
                    print('epoc: ', one_epoc, ', step: ', step, ', t_loss: ', t_cost, ', v_loss: ', v_cost)
            
            saver.save(sess, 'model/model_step2_%02d.ckpt' % one_epoc)
