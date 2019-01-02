# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 09:52:19 2018

@author: shen1994
"""

import os
# import numpy as np
import tensorflow as tf
# from tensorflow.python.framework.graph_util import convert_variables_to_constants
from tensorflow.python.tools import freeze_graph

if __name__ == "__main__":
    
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    
    saver = tf.train.import_meta_graph('model/model_step1_03.ckpt.meta') 

    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, 'model/model_step1_03.ckpt')
        
        #nodes = [node.name for node in sess.graph.as_graph_def().node]
        #for node in nodes:
        #    if node[0:9] == 'Stage2/la':
        #        print(node)
        
        #output_graph_def = convert_variables_to_constants(sess, 
        #                                                  tf.get_default_graph().as_graph_def(), 
        #                                                  output_node_names=["landmark"])
        #with tf.gfile.FastGFile('model/pico_FaceAlign_model.pb', mode='wb') as f:
        #    f.write(output_graph_def.SerializeToString())
        #'''
        
        tf.train.write_graph(sess.graph.as_graph_def(), 'model', 'model_graph.pb')
        freeze_graph.freeze_graph('model/model_graph.pb',
                                  '', 
                                  False, 
                                  'model/model_step1_03.ckpt', 
                                  'Stage2/landmark_1', 
                                  'save/restore_all', 
                                  'save/Const:0', 
                                  'model/pico_FaceAlign_model.pb', 
                                  False, 
                                  "")
