import logging
import os
import numpy as np
from numbers import Number

from . import tf
from phi import math
from phi.tf.flow import *


# --- Load Custom Ops ---
os.environ["CUDA_VISIBLE_DEVICES"]='0'
current_dir = os.path.dirname(os.path.realpath(__file__))
kernel_path = os.path.join(current_dir, 'cuda/build/quick_advection_op.so')
if not os.path.isfile(kernel_path):
        raise ImportError('CUDA binaries not found at %s. Run "python setup.py tf_cuda" to compile them' % kernel_path)
quick_op = tf.load_op_library(kernel_path)


def to_staggered_grid(data_x, data_y, dim):
    result_data = []
    for j in range(0, dim + 1):
        next = []
        for i in range(0, dim + 1):
            next.append([None, None])
        result_data.append(next)

    # X-Components (i+0.5,j)
    for j in range(0, dim):
        for i in range(0, dim + 1):
            result_data[j][i][1] = data_x[j][i]
    # Y-Components (i,j+0.5)
    for j in range(0, dim + 1):
        for i in range(0, dim):
            result_data[j][i][0] = data_y[j][i]

    return StaggeredGrid(np.array([result_data], dtype="float32"))    


def tf_cuda_quick_advection(velocity_field, dt, field=None, field_type="density", step_type="explicit_euler"):
    """
    Advects the field using the QUICK scheme
    :param field: The field to advect
    :param velocity_field: Velocity field for advection, equals to 'field' when velocity is advected
    :field_type: density, velovity
    :step_type: explicit_euler, adam_bashford
    :return: Advected field
    """

    if(field_type == "density"):
        density_tensor = tf.constant(field.data)
        density_tensor_padded = tf.constant(field.padded(2).data)
        velocity_v_field, velocity_u_field = velocity_field.data
        velocity_v_tensor = tf.constant(velocity_v_field.padded(2).data)
        velocity_u_tensor = tf.constant(velocity_u_field.padded(2).data)
        dimensions = field.data.shape[1]
        with tf.Session(""):
            result = quick_op.quick_advection(density_tensor, density_tensor_padded, velocity_u_tensor, velocity_v_tensor, dimensions, 2, dt, 0, 0).eval()
            return result

    elif(field_type == "velocity"):
        velocity_v_field, velocity_u_field = velocity_field.data
        velocity_v_tensor = tf.constant(velocity_v_field.data)
        velocity_u_tensor = tf.constant(velocity_u_field.data)
        velocity_v_tensor_padded = tf.constant(velocity_v_field.padded(2).data)
        velocity_u_tensor_padded = tf.constant(velocity_u_field.padded(2).data)
        dimensions = velocity_v_field.data.shape[1] - 1;
        with tf.Session(""):
            result_vel_u = quick_op.quick_advection(velocity_u_tensor, velocity_u_tensor_padded, velocity_u_tensor_padded, velocity_v_tensor_padded, dimensions, 2, dt, 1, 0).eval()
            result_vel_v = quick_op.quick_advection(velocity_v_tensor, velocity_v_tensor_padded, velocity_u_tensor_padded, velocity_v_tensor_padded, dimensions, 2, dt, 2, 0).eval()
            #return to_staggered_grid(velocity_u_field.data[0], result_vel_v[0], dimensions)
            #return to_staggered_grid(result_vel_u[0], velocity_v_field.data[0], dimensions)
            return to_staggered_grid(result_vel_u[0], result_vel_v[0], dimensions)

    print("QUICK Advection: Field type invalid!")
    return []
