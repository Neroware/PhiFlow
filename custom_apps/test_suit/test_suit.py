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

    np.random.seed(19680801)
    data = np.random.randn(30, 30)

    viridis = cm.get_cmap("viridis", 256)
    cms = [viridis]
   
    fig, axs = plt.subplots(1, 1, figsize=(3, 3), constrained_layout=True)
    for [ax, cmap] in zip([axs], cms):
        psm = ax.pcolormesh(data, cmap=cmap, rasterized=True, vmin=min_value, vmax=max_value)
        fig.colorbar(psm, ax=ax)
    fig.savefig("outputs/" + test_dir + "/" + dirname + "/" + filename)
    plt.close()


#def colorbar_to_image(min_value, max_value, dir_name, case_name, file_name, descr):
#    fig, ax = plt.subplots(figsize=(6, 1))
#    fig.subplots_adjust(bottom=0.5)
#
#    cmap = mpl.cm.cool
#    norm = mpl.colors.Normalize(vmin=min_value, vmax=max_value)
#
#    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='horizontal')
#    cb1.set_label(descr)
#    fig.savefig("outputs/" + dir_name + "/" + case_name + "/" + file_name + ".jpg")
    



#def array_to_image(arr, dirname, filename, min_value, max_value):
#    test_dir = "quick"
#    if semi_langrange_mode:
#        test_dir = "semi_lagrange"
#
#    try:
#        os.mkdir('outputs/' + test_dir + '/' + dirname)
#    except OSError:
#        pass
#    
#    img = []
#    for row in arr:
#        next = []
#        for col in row:
#            next.append(col[0])
#        img.append(next) 
#    plt.imsave('outputs/' + test_dir + '/' + dirname + '/' + filename, img, vmin=min_value, vmax=max_value)


def run_test_cases(test_cases):
    for test_case in test_cases:
        print("Starting Test Case '" + test_case.name + "'...")
        vel_min, vel_max = test_case.get_velocity_interval()
        den_min, den_max = test_case.get_density_interval()

        mode = "quick"
        if semi_langrange_mode:
            mode = "semi_lagrange"
        #colorbar_to_image(vel_min, vel_max, mode, test_case.name, "Velocity", "Velocity")
        #colorbar_to_image(den_min, den_max, mode, test_case.name, "Density", "Density")

        v_init = test_case.get_velocity_y()
        u_init = test_case.get_velocity_x()
        den_init = test_case.get_density()
        plot_grid(den_init[0], test_case.name, test_case.name + "_den_init.jpg", den_min, den_max)
        plot_grid(v_init[0], test_case.name, test_case.name + "_v_init.jpg", vel_min, vel_max)
        plot_grid(u_init[0], test_case.name, test_case.name + "_u_init.jpg", vel_min, vel_max)
        
        test_case.step()
        
        v_1 = test_case.get_velocity_y()
        u_1 = test_case.get_velocity_x()
        den_1 = test_case.get_density()
        plot_grid(den_1[0], test_case.name, test_case.name + "_den_1.jpg", den_min, den_max)
        plot_grid(v_1[0], test_case.name, test_case.name + "_v_1.jpg", vel_min, vel_max)
        plot_grid(u_1[0], test_case.name, test_case.name + "_u_1.jpg", vel_min, vel_max)

        for i in range(0, 100):
            test_case.step()

        v_100 = test_case.get_velocity_y()
        u_100 = test_case.get_velocity_x()
        den_100 = test_case.get_density()
        plot_grid(den_100[0], test_case.name, test_case.name + "_den_100.jpg", den_min, den_max)
        plot_grid(v_100[0], test_case.name, test_case.name + "_v_100.jpg", vel_min, vel_max)
        plot_grid(u_100[0], test_case.name, test_case.name + "_u_100.jpg", vel_min, vel_max)

        for i in range(0, 300):
            test_case.step()

        v_300 = test_case.get_velocity_y()
        u_300 = test_case.get_velocity_x()
        den_300 = test_case.get_density()
        plot_grid(den_300[0], test_case.name, test_case.name + "_den_300.jpg", den_min, den_max)
        plot_grid(v_300[0], test_case.name, test_case.name + "_v_300.jpg", vel_min, vel_max)
        plot_grid(u_300[0], test_case.name, test_case.name + "_u_300.jpg", vel_min, vel_max)

        print("Done!")


TEST_CASES = []


### Case 1 ###
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
        next.append([0.1 * math.sin(0.02 * PI * y), 0.1 * math.sin(0.02 * PI * x)])
    data.append(next)
velocity_array = np.array([data], dtype="float32")
velocity_field = StaggeredGrid(velocity_array)

if not semi_langrange_mode:
    case_1 = TestCase("Sin_xy", velocity_array, density_array, 0.1, vel_constant=True)
else:
    case_1 = TestCase("Sin_xy", velocity_field, density_field, 0.1, vel_constant=True)
TEST_CASES.append(case_1)


### Case 2 ###
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
        next.append([0.1 * math.sin(0.02 * PI * y), 0.1 * math.sin(0.02 * PI * x)])
    data.append(next)
velocity_array = np.array([data], dtype="float32")
velocity_field = StaggeredGrid(velocity_array)

if not semi_langrange_mode:
    case_2 = TestCase("Sin_xy_2", velocity_array, density_array, 0.1)
else:
    case_2 = TestCase("Sin_xy_2", velocity_field, density_field, 0.1)
TEST_CASES.append(case_2)


### Case 3 ###
data = []
for y in range(0, RESOLUTION[0]):
    next = []
    for x in range(0, RESOLUTION[0]):
        vx = x - 50
        vy = y - 50
        if(vx == 0.0 and vy == 0.0):
            vx = 1
            vy = 1
        m = math.sqrt(vx * vx + vy * vy)
        if(m > 6.0):
            m = 0.0
        else:
            m = 1.0
        next.append([0.3 * m])
    data.append(next)
density_array = np.array([data], dtype="float32")
density_field = CenteredGrid(density_array)

data = []
for y in range(0, RESOLUTION[0] + 1):
    next = []
    for x in range(0, RESOLUTION[0] + 1):
        next.append([0.005 * (y - 50), 0.005 * (x - 50)])
    data.append(next)
velocity_array = np.array([data], dtype="float32")
velocity_field = StaggeredGrid(velocity_array)

if not semi_langrange_mode:
    case_3 = TestCase("Escape_1", velocity_array, density_array, 0.1, vel_constant=True)
else:
    case_3 = TestCase("Escape_1", velocity_field, density_field, 0.1, vel_constant=True)
TEST_CASES.append(case_3)


### Case 4 ###
#data = []
#for y in range(0, RESOLUTION[0]):
#    next = []
#    for x in range(0, RESOLUTION[0]):
#        vx = x - 50
#        vy = y - 50
#        if(vx == 0.0 and vy == 0.0):
#            vx = 1
#            vy = 1
#        m = math.sqrt(vx * vx + vy * vy)
#        if(m > 6.0):
#            m = 0.0
#        else:
#            m = 1.0
#        next.append([0.3 * m])
#    data.append(next)
#density_array = np.array([data], dtype="float32")
#density_field = CenteredGrid(density_array)
#
#data = []
#for y in range(0, RESOLUTION[0] + 1):
#    next = []
#    for x in range(0, RESOLUTION[0] + 1):
#        next.append([0.005 * (y - 50), 0.005 * (x - 50)])
#    data.append(next)
#velocity_array = np.array([data], dtype="float32")
#velocity_field = StaggeredGrid(velocity_array)
#
#if not semi_langrange_mode:
#    case_4 = TestCase("Escape_2", velocity_array, density_array, 0.1)
#else:
#    case_4 = TestCase("Escape_2", velocity_field, density_field, 0.1)
#TEST_CASES.append(case_4)


### Case 5 ###
data = []
for y in range(0, RESOLUTION[0]):
    next = []
    for x in range(0, RESOLUTION[0]):
        vx = x - 50
        vy = y - 50
        if(vx == 0.0 and vy == 0.0):
            vx = 1
            vy = 1
        m = math.sqrt(vx * vx + vy * vy)
        if(m > 6.0):
            m = 0.0
        else:
            m = 1.0
        next.append([0.3 * m])
    data.append(next)
density_array = np.array([data], dtype="float32")
density_field = CenteredGrid(density_array)

data = []
for y in range(0, RESOLUTION[0] + 1):
    next = []
    for x in range(0, RESOLUTION[0] + 1):
        next.append([-0.2, 0.1])
    data.append(next)
velocity_array = np.array([data], dtype="float32")
velocity_field = StaggeredGrid(velocity_array)

if not semi_langrange_mode:
    case_5 = TestCase("Simple_Stream", velocity_array, density_array, 0.1)
else:
    case_5 = TestCase("Simple_Stream", velocity_field, density_field, 0.1)
TEST_CASES.append(case_5)

#cProfile.run('run_test_cases(TEST_CASES)')
run_test_cases(TEST_CASES)
