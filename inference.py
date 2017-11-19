#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 01:39:56 2017

@author: vishalp
"""

from __future__ import absolute_import, division, print_function
import os
import tensorflow as tf
import pandas as pd
from one_for_allmodel import multilayer_perceptron

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.app.flags.DEFINE_integer('compound' , '0', 'which compund to test on')
tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoints/','checkpoint directory to load the model from')
tf.app.flags.DEFINE_string('csv_file', 'toxicity_inference.csv', 'csv file for testing')

FLAGS = tf.app.flags.FLAGS

n_input = 1644
dropoutRate = tf.placeholder(tf.float32)
is_training= tf.placeholder(tf.bool)
markers = ['NR.AhR','NR.AR','NR.AR.LBD','NR.Aromatase','NR.ER','NR.ER.LBD','NR.PPAR.gamma','SR.ARE','SR.ATAD5','SR.HSE','SR.MMP','SR.p53']

def main(_):
    test_csv = pd.read_csv(FLAGS.csv_file, index_col = 0)
    compounds = test_csv.index.values
    print(compounds[FLAGS.compound])
    saver = tf.train.Saver()
    x = tf.placeholder("float", [None, n_input])
    x_in = test_csv.ix[compounds[FLAGS.compound]].values
    with tf.Session() as sess:
        pred = multilayer_perceptron(x, rate=dropoutRate, is_training=is_training)
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            pred_score = sess.run(pred, feed_dict={x:x_in.reshape(1,-1), dropoutRate:0.0, is_training:False})
            results_dict = dict(zip(markers, pred_score[0]))
            print (results_dict)

if __name__ == '__main__':
    tf.app.run()
