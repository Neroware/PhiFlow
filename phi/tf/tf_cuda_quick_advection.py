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


def tf_cuda_quick_advection(field, field_padded, vel_u, vel_v, dt, dim, field_type="density", step_type="explicit_euler"):
    """
    Advects the field using the QUICK scheme
    :param field:           Field tensor to advect
    :param field_padded     Padded Field tensor (padding=2)
    :param vel_u:           Padded Velocity u tensor
    :param vel_v:           Padded Velocity v tensor
    :param dt:              Timestep
    :param dim:             Dimension
    :param field_type:      density, velocity_u, velocity_v
    :param step_type:       explicit_euler (only euler supported)
    :return:                Advected field tensor
    """
    if(field_type == "density"):
        return quick_op.quick_advection(field, field_padded, vel_u, vel_v, dim, 2, dt, 0, 0)
    elif(field_type == "velocity_u"):
        return quick_op.quick_advection(field, field_padded, vel_u, vel_v, dim, 2, dt, 1, 0)
    elif(field_type == "velocity_v"):
        return quick_op.quick_advection(field, field_padded, vel_u, vel_v, dim, 2, dt, 2, 0)
    print("QUICK Advection: Field type invalid!")
    return None


@ops.RegisterGradient("QuickAdvection")
def _tf_cuda_quick_advection_grad(op, grad): 
    field = op.inputs[0]
    rho = op.inputs[1]
    u = op.inputs[2]
    v = op.inputs[3]
    
    paddings = tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]])
    grad_padded = tf.pad(grad, paddings)
    grad_x = quick_op.quick_advection(field, rho, u, v, 100, 2, 1.0, -1, 0)
    grad_y = quick_op.quick_advection(field, rho, u, v, 100, 2, 1.0, -2, 0)
    # Reverse Euler timestep made by OP
    grad_x = (grad_x - grad)
    grad_y = (grad_y - grad)

    # Resample here.... TODO
    x_pad = tf.constant([[0, 0], [2, 2], [3, 2], [0, 0]])
    y_pad = tf.constant([[0, 0], [3, 2], [2, 2], [0, 0]])
    grad_x = tf.pad(grad_x, x_pad)
    grad_y = tf.pad(grad_y, y_pad)

    #paddings = tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]])
    #grad_padded = tf.pad(grad, paddings, "CONSTANT")
    #res = quick_op.quick_advection(grad, grad_padded, u, v, 100, 2, 0.1, 0, 0)
    return [None, None, grad_x, grad_y]


def tf_cuda_quick_density_gradients(density, density_padded, vel_u, vel_v, dt, dim):
    rho_adv = quick_op.quick_advection(density, density_padded, vel_u, vel_v, dim, 2, dt, 0, 0)
    return tf.gradients(rho_adv, [vel_u, vel_v])



#def _tf_get_quick_coefficients(vel1, vel2):
#    def r_case0(): 
#        c1 = 0.125 * vel1
#        c2 = -0.125 * vel2 - 0.75 * vel1
#        c3 = 0.75 * vel2 - 0.375 * vel1
#        c4 = 0.375 * vel2
#        c5 = tf.constant(0.0)
#        return (c1, c2, c3, c4, c5)
#    def r_case1():
#        c1 = tf.constant(0.0)
#        c2 = -0.375 * vel1
#        c3 = 0.375 * vel2 - 0.75 * vel1
#        c4 = 0.75 * vel2 + 0.125 * vel1
#        c5 = -0.125 * vel2
#        return (c1, c2, c3, c4, c5)
#    def r_case2(): 
#        c1 = tf.constant(0.0)
#        c2 = -0.125 * vel2 - 0.375 * vel1
#        c3 = 0.75 * vel2 - 0.75 * vel1
#        c4 = 0.375 * vel2 + 0.125 * vel1
#        c5 = tf.constant(0.0)
#        return (c1, c2, c3, c4, c5)
#    def r_case3(): 
#        c1 = 0.125 * vel1
#        c2 = -0.75 * vel1
#        c3 = 0.375 * vel2 - 0.375 * vel1
#        c4 = 0.75 * vel2
#        c5 = -0.125 * vel1
#        return (c1, c2, c3, c4, c5)
#    def cond_1():
#        return tf.logical_and(tf.greater_equal(vel1, 0.0), tf.greater_equal(vel2, 0.0))
#    def cond_2():
#        return tf.logical_and(tf.less_equal(vel1, 0.0), tf.less_equal(vel2, 0.0))
#    def cond_3():
#        return tf.logical_and(tf.less(vel1, 0.0), tf.greater(vel2, 0.0))
#    return tf.cond(cond_1(), r_case0, lambda: tf.cond(cond_2(), r_case1, lambda: tf.cond(cond_3(), r_case2, r_case3)))


#def tf_quick_density_gradients(density_field, velocity_field, i, j, dt):
#    density = density_field.padded(2).data[0]
#    velocity = velocity_field.padded(2)
#    vel_v, vel_u = velocity.data
#
#    u1 = tf.constant(vel_u.data[0][j][i][0])
#    u2 = tf.constant(vel_u.data[0][j][i + 1][0])
#    v1 = tf.constant(vel_v.data[0][j][i][0])
#    v2 = tf.constant(vel_v.data[0][j + 1][i][0])
#
#    v_c1, v_c2, v_c3, v_c4, v_c5 = _tf_get_quick_coefficients(v1, v2)
#    u_c1, u_c2, u_c3, u_c4, u_c5 = _tf_get_quick_coefficients(u1, u2)
#
#    rho_x = []
#    rho_y = []
#    for offset in range(0, 5):
#        rho_x.append(tf.constant(density[j + 2][i + offset][0]))
#        rho_y.append(tf.constant(density[j + offset][i + 2][0]))
#
#    next_rho_x = rho_x[2] + (u_c1 * rho_x[0] + u_c2 * rho_x[1] + u_c3 * rho_x[2] + u_c4 * rho_x[3] + u_c5 * rho_x[4]) * dt
#    next_rho_y = rho_y[2] + (v_c1 * rho_y[0] + v_c2 * rho_y[1] + v_c3 * rho_y[2] + v_c4 * rho_y[3] + v_c5 * rho_y[4]) * dt
#
#    return tf.gradients(next_rho_x, [u1, u2]), tf.gradients(next_rho_y, [v1, v2])
