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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PI = 3.14159

semi_langrange_mode = False
if not 'quick' in sys.argv:
    semi_langrange_mode = True



class TestCase:
    def __init__(self, name, velocity_field, density_field, timestep, vel_constant=False):
        self.name = name
        self.velocity_field = velocity_field
        self.density_field = density_field
        self.timestep = timestep
        self.vel_constant = vel_constant


    def step(self): 
        if not semi_langrange_mode:
            try:
                velocity = StaggeredGrid(self.velocity_field)
            except:
                velocity = self.velocity_field
            density = CenteredGrid(self.density_field)
            dt = self.timestep            
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




def array_to_image(arr, dirname, filename):
    test_dir = "quick"
    if semi_langrange_mode:
        test_dir = "semi_lagrange"

    try:
        os.mkdir('outputs/' + test_dir + '/' + dirname)
    except OSError:
        pass
    
    img = []
    for row in arr:
        next = []
        for col in row:
            next.append(col[0])
        img.append(next) 
    plt.imsave('outputs/' + test_dir + '/' + dirname + '/' + filename, img)


def run_test_cases(test_cases):
    for test_case in test_cases:
        print("Starting Test Case '" + test_case.name + "'...")
        v_init = test_case.get_velocity_y()
        #print(">>>>> ", v_init)
        u_init = test_case.get_velocity_x()
        den_init = test_case.get_density()
        array_to_image(den_init[0], test_case.name, test_case.name + "_den_init.jpg")
        array_to_image(v_init[0], test_case.name, test_case.name + "_v_init.jpg")
        array_to_image(u_init[0], test_case.name, test_case.name + "_u_init.jpg")

        test_case.step()
        
        v_1 = test_case.get_velocity_y()
        #print("-----> ", v_1)
        u_1 = test_case.get_velocity_x()
        den_1 = test_case.get_density()
        array_to_image(den_1[0], test_case.name, test_case.name + "_den_1.jpg")
        array_to_image(v_1[0], test_case.name, test_case.name + "_v_1.jpg")
        array_to_image(u_1[0], test_case.name, test_case.name + "_u_1.jpg")

        for i in range(0, 100):
            test_case.step()

        v_100 = test_case.get_velocity_y()
        u_100 = test_case.get_velocity_x()
        den_100 = test_case.get_density()
        array_to_image(den_100[0], test_case.name, test_case.name + "_den_100.jpg")
        array_to_image(v_100[0], test_case.name, test_case.name + "_v_100.jpg")
        array_to_image(u_100[0], test_case.name, test_case.name + "_u_100.jpg")

        for i in range(0, 300):
            test_case.step()

        v_300 = test_case.get_velocity_y()
        u_300 = test_case.get_velocity_x()
        den_300 = test_case.get_density()
        array_to_image(den_300[0], test_case.name, test_case.name + "_den_300.jpg")
        array_to_image(v_300[0], test_case.name, test_case.name + "_v_300.jpg")
        array_to_image(u_300[0], test_case.name, test_case.name + "_u_300.jpg")

        print("Done!")


TEST_CASES = []


### Case 1 ###
data = []
for y in range(0, RESOLUTION[0]):
    next = []
    for x in range(0, RESOLUTION[0]):
        if x % 8 <= 3 and y % 8 <= 3:
            next.append([0.5])
        elif x % 8 > 3 and y % 8 <= 3:
            next.append([1.0])
        elif x % 8 <= 3 and y % 8 > 3:
            next.append([1.0])
        else:
            next.append([0.5])
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


### Test 2 ###
velocity_array = np.array([data], dtype="float32")
velocity_field = StaggeredGrid(velocity_array)
density_array = np.array([data], dtype="float32")
density_field = CenteredGrid(density_array)
if not semi_langrange_mode:
    case_2 = TestCase("Sin_xy_2", velocity_array, density_array, 0.1)
else:
    case_2 = TestCase("Sin_xy_2", velocity_field, density_field, 0.1)
TEST_CASES.append(case_2)




run_test_cases(TEST_CASES)

