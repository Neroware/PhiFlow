import logging
import os
import numpy as np
from numbers import Number

from . import tf
from phi import math


# --- Check if TF enabled (remove later!)  ---
#import sys
#if not 'tf' in sys.argv:
#   raise RuntimeError("QUICK Advection: This module can only be run in TF mode!")

# --- Load Custom Ops ---
current_dir = os.path.dirname(os.path.realpath(__file__))
kernel_path = os.path.join(current_dir, 'cuda/build/quick_advection_op.so')
if not os.path.isfile(kernel_path):
        raise ImportError('CUDA binaries not found at %s. Run "python setup.py tf_cuda" to compile them' % kernel_path)
quick_op = tf.load_op_library(kernel_path)


def tf_cuda_quick_advection(field, velocity_field, dt, field_type="density", step_type="explicit_euler"):
    """
    Advects the field using the QUICK scheme
    :param field: The field to advect
    :param velocity_field: Velocity field for advection, equals to 'field' when velocity is advected
    :field_type: density, velocity
    :step_type: explicit_euler, adam_bashford
    :return: Advected field
    """

    if(field_type == "density"):
        dimensions = field.data.shape[1]
        density_tensor = field.data
        velocity_tensor = velocity_field.data

        with tf.Session(""):
            print(">>>>>>>>>>>>>>>> ", type(density_tensor))
            quick_op.quick_advection(density_tensor, velocity_tensor).eval()
        


def test_cuda():
    print("Sarting CUDA TF test...")
    with tf.Session(""):
        quick_op.quick_advection([[1, 2], [3, 4]]).eval()
    #o = quick_op.quick_advection(24)
    print("RESULT OF CUDA TEST")

