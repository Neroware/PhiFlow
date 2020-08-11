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


def _tf_get_quick_coefficients(vel1, vel2):
    def r_case0(): 
        c1 = tf.constant(0.125 * vel1)
        c2 = tf.constant(-0.125 * vel2 - 0.75 * vel1)
        c3 = tf.constant(0.75 * vel2 - 0.375 * vel1)
        c4 = tf.constant(0.375 * vel2)
        c5 = tf.constant(0.0)
        return (c1, c2, c3, c4, c5)
    def r_case1():
        c1 = tf.constant(0.0)
        c2 = tf.constant(-0.375 * vel1)
        c3 = tf.constant(0.375 * vel2 - 0.75 * vel1)
        c4 = tf.constant(0.75 * vel2 + 0.125 * vel1)
        c5 = tf.constant(-0.125 * vel2)
        return (c1, c2, c3, c4, c5)
    def r_case2(): 
        c1 = tf.constant(0.0)
        c2 = tf.constant(-0.125 * vel2 - 0.375 * vel1)
        c3 = tf.constant(0.75 * vel2 - 0.75 * vel1)
        c4 = tf.constant(0.375 * vel2 + 0.125 * vel1)
        c5 = tf.constant(0.0)
        return (c1, c2, c3, c4, c5)
    def r_case3(): 
        c1 = tf.constant(0.125 * vel1)
        c2 = tf.constant(-0.75 * vel1)
        c3 = tf.constant(0.375 * vel2 - 0.375 * vel1)
        c4 = tf.constant(0.75 * vel2)
        c5 = tf.constant(-0.125 * vel1)
        return (c1, c2, c3, c4, c5)
    def cond_1():
        return tf.logical_and(tf.greater_equal(vel1, 0.0), tf.greater_equal(vel2, 0.0))
    def cond_2():
        return tf.logical_and(tf.lower_equal(vel1, 0.0), tf.lower_equal(vel2, 0.0))
    def cond_3():
        return tf.logical_and(tf.lower(vel1, 0.0), tf.greater(vel2, 0.0))
    return tf.cond(cond_1(), r_case0, tf.cond(cond_2(), r_case1, tf.cond(cond_3(), r_case2, r_case3))


#def _get_quick_coefficients(vel1, vel2):
#    c1 = c2 = c3 = c4 = c5 = 0.0
#    if(vel1 >= 0 and vel2 >= 0):
#        c1 = 0.125 * vel1
#        c2 = -0.125 * vel2 - 0.75 * vel1
#        c3 = 0.75 * vel2 - 0.375 * vel1
#        c4 = 0.375 * vel2
#    elif(vel1 <= 0 and vel2 <= 0):
#        c2 = -0.375 * vel1
#        c3 = 0.375 * vel2 - 0.75 * vel1
#        c4 = 0.75 * vel2 + 0.125 * vel1
#        c5 = -0.125 * vel2
#    elif(vel1 < 0 and vel2 > 0):
#        c2 = -0.125 * vel2 - 0.375 * vel1
#        c3 = 0.75 * vel2 - 0.75 * vel1 
#        c4 = 0.375 * vel2 + 0.125 * vel1
#    else:
#        c1 = 0.125 * vel1
#        c2 = -0.75 * vel1
#        c3 = 0.375 * vel2 - 0.375 * vel1
#        c4 = 0.75 * vel2
#        c5 = -0.125 * vel1
#    return (c1, c2, c3, c4, c5)


def tf_quick_advection_coefficients(velocity_field, i, j, dt):
    """ 
    Returns the coefficients needed to calculate rho(t+1) at (i,j)
    :param velocity_field: Staggered Grid with velocity entries
    :param i:              Density Cell X-coordinate
    :param j:              Density Cell Y-coordinate
    :param dt:             Timestep
    :return:               v coefficients, u coefficients; Ranging from (j-2,i) to (j+2,i) and (i-2,j) to (i+2,j)
    """
    velocity_v_field, velocity_u_field = velocity_field.data
    u1 = tf.constant(velocity_u_field.data[0][j][i][0])
    u2 = tf.constant(velocity_u_field.data[0][j][i + 1][0])
    v1 = tf.constant(velocity_v_field.data[0][j][i][0])
    v2 = tf.constant(velocity_v_field.data[0][j + 1][i][0])
    u_coefficients = _tf_get_quick_coefficients(u1, u2)
    v_coefficients = _tf_get_quick_coefficients(v1, v2)
    return (v_coefficients, u_coefficients)


def tf_quick_density_gradients(density_field, velocity_field, i, j, dt):
    density = density_field.padded(2).data[0]
    velocity = velocity_field.padded(2)
    vel_v, vel_u = velocity.data

    u1 = tf.constant(vel_u.data[0][j][i][0])
    u2 = tf.constant(vel_u.data[0][j][i + 1][0])
    v1 = tf.constant(vel_v.data[0][j][i][0])
    v2 = tf.constant(vel_v.data[0][j + 1][i][0])

    v_c1, v_c2, v_c3, v_c4, v_c5 = _get_quick_coefficients(v1, v2)
    u_c1, u_c2, u_c3, u_c4, u_c5 = _get_quick_coefficients(u1, u2)

    rho_x = []
    rho_y = []
    for offset in range(0, 5):
        rho_x.append(tf.constant(density[j + 2][i + offset][0]))
        rho_y.append(tf.constant(density[j + offset][i + 2][0]))

    next_rho_x = rho_x[2] + (u_c1 * rho_x[0] + u_c2 * rho_x[1] + u_c3 * rho_x[2] + u_c4 * rho_x[3] + u_c5 * rho_x[4]) * dt
    next_rho_y = rho_y[2] + (v_c1 * rho_y[0] + v_c2 * rho_y[1] + v_c3 * rho_y[2] + v_c4 * rho_y[3] + v_c5 * rho_y[4]) * dt

    return tf.gradients(next_rho_x, [u1, u2, v1, v2]), tf.gradients(next_rho_y, [u1, u2, v1, v2])


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
