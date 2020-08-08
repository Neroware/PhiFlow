import logging
import os
import numpy as np
from numbers import Number

from . import tf
from phi import math
from phi.tf.flow import *

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops


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
    :param velocity_field: Velocity field for advection, equals to 'field' when velocity is advected
    :param dt:             Timestep
    :param field:          Field to advect
    :param field_type:     density, velocity
    :param step_type:      explicit_euler (only euler supported)
    :return:               Advected field
    """
    if(field_type == "density"):
        density_tensor = tf.constant(field.data)
        density_tensor_padded = tf.constant(field.padded(2).data)
        velocity_v_field, velocity_u_field = velocity_field.data
        velocity_v_tensor = tf.constant(velocity_v_field.padded(2).data)
        velocity_u_tensor = tf.constant(velocity_u_field.padded(2).data)
        dimensions = field.data.shape[1]
        with tf.compat.v1.Session("") as sess:
            result = quick_op.quick_advection(density_tensor, density_tensor_padded, velocity_u_tensor, velocity_v_tensor, dimensions, 2, dt, 0, 0).eval()
            return result
    elif(field_type == "velocity"):
        velocity_v_field, velocity_u_field = velocity_field.data
        velocity_v_tensor = tf.constant(velocity_v_field.data)
        velocity_u_tensor = tf.constant(velocity_u_field.data)
        velocity_v_tensor_padded = tf.constant(velocity_v_field.padded(2).data)
        velocity_u_tensor_padded = tf.constant(velocity_u_field.padded(2).data)
        dimensions = velocity_v_field.data.shape[1] - 1;
        with tf.compat.v1.Session(""):
            result_vel_u = quick_op.quick_advection(velocity_u_tensor, velocity_u_tensor_padded, velocity_u_tensor_padded, velocity_v_tensor_padded, dimensions, 2, dt, 1, 0).eval()
            result_vel_v = quick_op.quick_advection(velocity_v_tensor, velocity_v_tensor_padded, velocity_u_tensor_padded, velocity_v_tensor_padded, dimensions, 2, dt, 2, 0).eval()
            return to_staggered_grid(result_vel_u[0], result_vel_v[0], dimensions)
    print("QUICK Advection: Field type invalid!")
    return []


def tf_quick_advection_coefficients(velocity_field, i, j, dt):
    """ 
    Returns the coefficients needed to calculate rho(t+1) at (i,j)
    :param velocity_field: Staggered Grid with velocity entries
    :param i:              Density Cell X-coordinate
    :param j:              Density Cell Y-coordinate
    :param dt:             Timestep
    :return:               v coefficients, u coefficients; Ranging from (j-2,i) to (j+2,i) and (i-2,j) to (i+2,j)
    """
    def coefficients(vel1, vel2):
        c1 = c2 = c3 = c4 = c5 = 0.0
        if(vel1 >= 0 and vel2 >= 0):
            c1 = 0.125 * vel1
            c2 = -0.125 * vel2 - 0.75 * vel1
            c3 = 0.75 * vel2 - 0.375 * vel1
            c4 = 0.375 * vel2
        elif(vel1 <= 0 and vel2 <= 0):
            c2 = -0.375 * vel1
            c3 = 0.375 * vel2 - 0.75 * vel1
            c4 = 0.75 * vel2 + 0.125 * vel1
            c5 = -0.125 * vel2
        elif(vel1 < 0 and vel2 > 0):
            c2 = -0.125 * vel2 - 0.375 * vel1
            c3 = 0.75 * vel2 - 0.75 * vel1 
            c4 = 0.375 * vel2 + 0.125 * vel1
        else:
            c1 = 0.125 * vel1
            c2 = -0.75 * vel1
            c3 = 0.375 * vel2 - 0.375 * vel1
            c4 = 0.75 * vel2
            c5 = -0.125 * vel1
        return (c1, c2, c3, c4, c5)
    velocity_v_field, velocity_u_field = velocity_field.data
    u1 = velocity_u_field.data[j][i]
    u2 = velocity_u_field.data[j][i + 1]
    v1 = velocity_v_field.data[j][i]
    v2 = velocity_v_field.data[j + 1][i]
    u_coefficients = coefficients(u1, u2)
    v_coefficients = coefficients(v1, v2)
    return (v_coefficients, u_coefficients)


#@ops.RegisterGradient("QuickAdvection")
#def _tf_quick_advection_gradients(op, grad):
#    result = [None, None, None, None]
#    field = op.inputs[0]
#    rho0 = op.inputs[1]
#    u = op.inputs[2]
#    v = op.inputs[3]
#    return [field, rho0, u, v]


#def tf_quick_density_gradients(density_field, velocity_field, dt):
#    dimensions = density_field.data.shape[1]
#    field_tensor = tf.constant(density_field.data)
#    rho0 = tf.constant(density_field.padded(2).data)
#    u = tf.constant(velocity_field.data[0].padded(2).data)
#    v = tf.constant(velocity_field.data[1].padded(2).data)
#    rho1 = quick_op.quick_advection(field_tensor, rho0, u, v, dimensions, 2, dt, 0, 0)
#    return tf.gradients(rho1, [field_tensor, rho0, u, v])


#from phi.physics.field.advect import semi_lagrangian
#def tf_semi_lagrange_density_gradients(density_field, velocity_field, dt):
#    dimensions = density_field.data.shape[1]
#
#    def semi_lagrange_adv(rho0, u, v):
#        with tf.compat.v1.Session(""):
#            rho0_data = rho0.eval()
#            u_data = u.eval()
#            v_data = v.eval()
#            rho0_grid = CenteredGrid(rho0_data)
#            vel_grid = to_staggered_grid(u_data[0], v_data[0], dimensions)
#            return tf.constant(semi_lagrangian(rho0_grid, vel_grid, dt).data[0])
#
#    rho0 = tf.constant(density_field.padded(2).data)
#    v = tf.constant(velocity_field.data[0].padded(2).data)
#    u = tf.constant(velocity_field.data[1].padded(2).data)
#    rho1 = semi_lagrange_adv(rho0, u, v)
#    return tf.gradients(rho1, [rho0, u, v])
