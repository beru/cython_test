#!/usr/bin/python3

import adder
import numpy as np
import ctypes

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

def test():
    t = tf.zeros([10, 5])
    print(t)
    packed = tf.experimental.dlpack.to_dlpack(t)
    adder.add(packed, 111)
    t = tf.experimental.dlpack.from_dlpack(packed)
    print(t)

test()
