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
os.environ["CUDA_VISIBLE_DEVICES"]='1'
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
        density_tensor = tf.constant(field.data)
        velocity_v_field, velocity_u_field = velocity_field.data
        velocity_v_tensor = tf.constant(velocity_v_field.data)
        velocity_u_tensor = tf.constant(velocity_u_field.data)
        dimensions = field.data.shape[1]
        with tf.Session(""):
            result = quick_op.quick_advection(density_tensor, velocity_u_tensor, velocity_v_tensor, dimensions, dt, 0, 0).eval()
            print("======> ", result)
            return result
