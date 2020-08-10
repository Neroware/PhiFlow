import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
if not 'tf' in sys.argv:
    raise RuntimeError("This simulation can only be run in TensorFlow-Mode!")
from phi.tf.flow import *  # Use TensorFlow
MODE = 'TensorFlow'
RESOLUTION = [int(sys.argv[1])] * 2 if len(sys.argv) > 1 and __name__ == '__main__' else [128] * 2
DESCRIPTION = "Basic fluid test that runs QUICK Scheme with CUDA"

import numpy as np

from phi.tf.tf_cuda_quick_advection import tf_cuda_quick_advection
from phi.physics.field.advect import semi_lagrangian

import math

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

#import cProfile




PI = 3.14159

semi_langrange_mode = False
if not 'quick' in sys.argv:
    semi_langrange_mode = True



class TestCase:
    def __init__(self, name, velocity_field, density_field, timestep, vel_constant=False, den_interval=(-0.1, 0.4), vel_interval=(-0.4, 0.4)):
        self.name = name
        self.velocity_field = velocity_field
        self.density_field = density_field
        self.timestep = timestep
        self.vel_constant = vel_constant
        self.den_interval = den_interval
        self.vel_interval = vel_interval


    def step(self):
        if not semi_langrange_mode:
            try:
                velocity = StaggeredGrid(self.velocity_field)
            except:
                velocity = self.velocity_field
            density = CenteredGrid(self.density_field)
            dt = self.timestep
            tf.compat.v1.reset_default_graph()
            self.density_field = tf_cuda_quick_advection(velocity, dt, field=density, field_type="density")
            if not self.vel_constant:
                self.velocity_field = tf_cuda_quick_advection(velocity, dt, field_type="velocity")
        else:
            velocity = self.velocity_field
            density = self.density_field
            dt = self.timestep
            self.density_field = advect.semi_lagrangian(density, velocity, dt=dt)
            if not self.vel_constant:
                self.velocity_field = advect.semi_lagrangian(velocity, velocity, dt=dt)


    def get_velocity_y(self):
        if(semi_langrange_mode):
            return self.velocity_field.data[0].data

        if str(type(self.velocity_field.data)) == "<class 'tuple'>":
             return self.velocity_field.data[0].data
        
        data = np.array(self.velocity_field.data)
        arr = []
        for row in data[0]:
            next = []
            for col in row:
                next.append([col[0]])
            arr.append(next)
        return np.array([arr])


    def get_velocity_x(self):
        if(semi_langrange_mode):
            return self.velocity_field.data[1].data

        if str(type(self.velocity_field.data)) == "<class 'tuple'>":
            return self.velocity_field.data[1].data

        data = np.array(self.velocity_field.data)
        arr = []
        for row in data[0]:
            next = []
            for col in row:
                next.append([col[1]])
            arr.append(next)
        return np.array([arr])


    def get_density(self):
        return np.array(self.density_field.data)


    def get_density_interval(self):
        return self.den_interval


    def get_velocity_interval(self):
        return self.vel_interval


def plot_grid(data, dirname, filename, min_value, max_value):
    test_dir = "quick"
    if semi_langrange_mode:
        test_dir = "semi_lagrange"

    img = []
    for row in data:
        next = []
        for col in row:
            next.append(col[0])
        img.append(next)

    viridis = cm.get_cmap("viridis", 256)
    cms = [viridis]
   
    fig, axs = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
    for [ax, cmap] in zip([axs], cms):
        psm = ax.pcolormesh(img, cmap=cmap, rasterized=True, vmin=min_value, vmax=max_value)
        fig.colorbar(psm, ax=ax)
    fig.savefig("outputs/" + test_dir + "/" + dirname + "/" + filename)
    plt.close()
    

def plot_grid_diff_y_mirror(data1, data2, res, filename):
    diff = []
    for j in range(0, res):
        next = []
        for i in range(0, res + 1):
            next.append([-0.4])
        diff.append(next)

    for j in range(0, res):
         for i in range(0, res + 1):
             diff[j][i][0] = data2[j][i][0] - data1[res - j - 1][i][0]

    plot_grid(diff, "diff", filename, -0.4, 0.4)



def run_test_cases(test_cases):
    case_1 = test_cases[0]
    case_2 = test_cases[1]
    vel_min, vel_max = case_1.get_velocity_interval()
    den_min, den_max = case_1.get_density_interval()

    mode = "quick"
    if semi_langrange_mode:
        mode = "semi_lagrange"

    v1_init = case_1.get_velocity_y()
    u1_init = case_1.get_velocity_x()
    den1_init = case_1.get_density()
    v2_init = case_2.get_velocity_y()
    u2_init = case_2.get_velocity_x()
    den2_init = case_2.get_density()
    #plot_grid_diff_y_mirror(u1_init[0], u2_init[0], RESOLUTION[0], "diff_init.jpg")

    case_1.step()
    case_2.step()

    v1_1 = case_1.get_velocity_y()
    u1_1 = case_1.get_velocity_x()
    den1_1 = case_1.get_density()
    v2_1 = case_2.get_velocity_y()
    u2_1 = case_2.get_velocity_x()
    den2_1 = case_2.get_density()
    plot_grid_diff_y_mirror(u1_1[0], u2_1[0], RESOLUTION[0], "diff_1.jpg")

    for count in range(0, 6):
        for step in range(0, 50):
            case_1.step()
            case_2.step()
        v1 = case_1.get_velocity_y()
        u1 = case_1.get_velocity_x()
        den1 = case_1.get_density()
        v2 = case_2.get_velocity_y()
        u2 = case_2.get_velocity_x()
        den2 = case_2.get_density()
        plot_grid_diff_y_mirror(u1[0], u2[0], RESOLUTION[0], "diff_" + str((count + 1) * 50) + ".jpg")


TEST_CASES = []


### Case 7a ###
data = []
for y in range(0, RESOLUTION[0]):
    next = []
    for x in range(0, RESOLUTION[0]):
        if x % 8 <= 3 and y % 8 <= 3:
            next.append([0.1])
        elif x % 8 > 3 and y % 8 <= 3:
            next.append([0.2])
        elif x % 8 <= 3 and y % 8 > 3:
            next.append([0.2])
        else:
            next.append([0.1])
    data.append(next)
density_array = np.array([data], dtype="float32")
density_field = CenteredGrid(density_array)

data = []
for y in range(0, RESOLUTION[0] + 1):
    next = []
    for x in range(0, RESOLUTION[0] + 1):
        if(x >= 45 and x < 55 and y >= 45 and y < 55):
            next.append([0.1, 0.2])
        else:
            next.append([0.1, 0.1])
    data.append(next)
velocity_array = np.array([data], dtype="float32")
velocity_field = StaggeredGrid(velocity_array)

if not semi_langrange_mode:
    case_7a = TestCase("Vel_X_Block", velocity_array, density_array, 0.1)
else:
    case_7a = TestCase("Vel_X_Block", velocity_field, density_field, 0.1)
TEST_CASES.append(case_7a)


### Case 7b ###
data = []
for y in range(0, RESOLUTION[0]):
    next = []
    for x in range(0, RESOLUTION[0]):
        if x % 8 <= 3 and y % 8 <= 3:
            next.append([0.1])
        elif x % 8 > 3 and y % 8 <= 3:
            next.append([0.2])
        elif x % 8 <= 3 and y % 8 > 3:
            next.append([0.2])
        else:
            next.append([0.1])
    data.append(next)
density_array = np.array([data], dtype="float32")
density_field = CenteredGrid(density_array)

data = []
for y in range(0, RESOLUTION[0] + 1):
    next = []
    for x in range(0, RESOLUTION[0] + 1):
        if(x >= 45 and x < 55 and y >= 45 and y < 55):
            next.append([-0.1, 0.2])
        else:
            next.append([-0.1, 0.1])
    data.append(next)
velocity_array = np.array([data], dtype="float32")
velocity_field = StaggeredGrid(velocity_array)

if not semi_langrange_mode:
    case_7b = TestCase("Vel_X_Block_2", velocity_array, density_array, 0.1)
else:
    case_7b = TestCase("Vel_X_Block_2", velocity_field, density_field, 0.1)
TEST_CASES.append(case_7b)




#cProfile.run('run_test_cases(TEST_CASES)')
run_test_cases(TEST_CASES)
