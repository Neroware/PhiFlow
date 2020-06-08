import logging
import os
import numpy as np
from numbers import Number

from . import tf
from phi import math


# --- Load Custom Ops ---
current_dir = os.path.dirname(os.path.realpath(__file__))
kernel_path = os.path.join(current_dir, 'cuda/build/quick_advection_op.so')
if not os.path.isfile(kernel_path):
        raise ImportError('CUDA binaries not found at %s. Run "python setup.py tf_cuda" to compile them' % kernel_path)
quick_op = tf.load_op_library(kernel_path)


def test_cuda():
    print("Sarting CUDA TF test...")
    with tf.Session(""):
        quick_op.quick_advection([[1, 2], [3, 4]]).eval()

    #o = quick_op.quick_advection(24)
    print("RESULT OF CUDA TEST")
