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
kernel_path = os.path.join(current_dir, 'cuda/build/quick_advection_op_gradient.so')
if not os.path.isfile(kernel_path):
        raise ImportError('CUDA binaries not found at %s. Run "python setup.py tf_cuda" to compile them' % kernel_path)
quick_op_gradient = tf.load_op_library(kernel_path)



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
    dimensions = op.get_attr("dimensions")
    timestep = op.get_attr("timestep")
    padding = op.get_attr("padding")
    grad_rho, grad_u, grad_v = quick_op_gradient.quick_advection_gradient(field, rho, u, v, dimensions, padding, timestep)
    return [None, grad_rho, grad_u, grad_v]


def tf_cuda_quick_density_gradients(density, density_padded, vel_u, vel_v, dt, dim):
    rho_adv = quick_op.quick_advection(density, density_padded, vel_u, vel_v, dim, 2, dt, 0, 0)
    return tf.gradients(rho_adv, [density_padded, vel_u, vel_v])
