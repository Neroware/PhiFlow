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

from phi.tf.tf_cuda_quick_advection import *
from phi.physics.field.advect import semi_lagrangian

import math

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

#import cProfile




PI = 3.14159
EXP0 = 2.71828

semi_langrange_mode = False
if not 'quick' in sys.argv:
    semi_langrange_mode = True

DT = 10.0


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


class TestCase:
    def __init__(self, name, velocity_field, density_field, timestep, vel_constant=False, den_interval=(-0.1, 0.4), vel_interval=(-0.4, 0.4)):
        self.name = name
        self.velocity_field = velocity_field
        self.density_field = density_field
        self.timestep = timestep
        self.vel_constant = vel_constant
        self.den_interval = den_interval
        self.vel_interval = vel_interval


    def save_gradients(self, filename_postfix):
        if not semi_langrange_mode:
            try:
                velocity = StaggeredGrid(self.velocity_field)
            except:
                velocity = self.velocity_field
            try:
                density = CenteredGrid(self.density_field)
            except:
                density = self.density_field
            dt = self.timestep
            dim = RESOLUTION[0]
            tf.compat.v1.reset_default_graph()
        
            density_tensor = tf.constant(density.data)
            density_tensor_padded = tf.constant(density.padded(2).data)
            velocity_v_field, velocity_u_field = velocity.data
            velocity_v_tensor = tf.constant(velocity_v_field.data)
            velocity_u_tensor = tf.constant(velocity_u_field.data)
            velocity_v_tensor_padded = tf.constant(velocity_v_field.padded(2).data)
            velocity_u_tensor_padded = tf.constant(velocity_u_field.padded(2).data)

            with tf.Session("") as sess:
                grd_field, grd_u, grd_v = tf_cuda_quick_density_gradients(density_tensor, density_tensor_padded, velocity_u_tensor_padded, velocity_v_tensor_padded, dt, dim)
                plot_grid(grd_field.eval()[0], self.name, self.name + "_den_grad" + filename_postfix + ".jpg", -0.04, 0.04)
                plot_grid(grd_u.eval()[0], self.name, self.name + "_u_grad" + filename_postfix + ".jpg", -0.04, 0.04)
                plot_grid(grd_v.eval()[0], self.name, self.name + "_v_grad" + filename_postfix + ".jpg", -0.04, 0.04)
                sess.close()


    def step(self):
        if not semi_langrange_mode:
            try:
                velocity = StaggeredGrid(self.velocity_field)
            except:
                velocity = self.velocity_field
            try:
                density = CenteredGrid(self.density_field)
            except:
                density = self.density_field
            dt = self.timestep
            dim = RESOLUTION[0]
            tf.compat.v1.reset_default_graph()

            density_tensor = tf.constant(density.data)
            density_tensor_padded = tf.constant(density.padded(2).data)
            velocity_v_field, velocity_u_field = velocity.data
            velocity_v_tensor = tf.constant(velocity_v_field.data)
            velocity_u_tensor = tf.constant(velocity_u_field.data)
            velocity_v_tensor_padded = tf.constant(velocity_v_field.padded(2).data)
            velocity_u_tensor_padded = tf.constant(velocity_u_field.padded(2).data)

            den = tf_cuda_quick_advection(density_tensor, density_tensor_padded, velocity_u_tensor_padded, velocity_v_tensor_padded, dt, dim, field_type="density")
            vel_u = tf_cuda_quick_advection(velocity_u_tensor, velocity_u_tensor_padded, velocity_u_tensor_padded, velocity_v_tensor_padded, dt, dim, field_type="velocity_u")
            vel_v = tf_cuda_quick_advection(velocity_v_tensor, velocity_v_tensor_padded, velocity_u_tensor_padded, velocity_v_tensor_padded, dt, dim, field_type="velocity_v")

            with tf.Session("") as sess:
                self.density_field = CenteredGrid(den.eval())
                self.velocity_field = to_staggered_grid(vel_u.eval()[0], vel_v.eval()[0], dim)
                sess.close()

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

        test_case.save_gradients("_init")
        #test_case.step()
        
        #v_1 = test_case.get_velocity_y()
        #u_1 = test_case.get_velocity_x()
        #den_1 = test_case.get_density()
        #plot_grid(den_1[0], test_case.name, test_case.name + "_den_1.jpg", den_min, den_max)
        #plot_grid(v_1[0], test_case.name, test_case.name + "_v_1.jpg", vel_min, vel_max)
        #plot_grid(u_1[0], test_case.name, test_case.name + "_u_1.jpg", vel_min, vel_max)

        for i in range(0, int(100.0 * (0.1 / DT))):
            test_case.step()
        test_case.save_gradients("_100")

        v_100 = test_case.get_velocity_y()
        u_100 = test_case.get_velocity_x()
        den_100 = test_case.get_density()
        plot_grid(den_100[0], test_case.name, test_case.name + "_den_100.jpg", den_min, den_max)
        plot_grid(v_100[0], test_case.name, test_case.name + "_v_100.jpg", vel_min, vel_max)
        plot_grid(u_100[0], test_case.name, test_case.name + "_u_100.jpg", vel_min, vel_max)

        for i in range(0, int(300.0 * (0.1 / DT))):
            test_case.step()
        test_case.save_gradients("_400")

        v_400 = test_case.get_velocity_y()
        u_400 = test_case.get_velocity_x()
        den_400 = test_case.get_density()
        plot_grid(den_400[0], test_case.name, test_case.name + "_den_400.jpg", den_min, den_max)
        plot_grid(v_400[0], test_case.name, test_case.name + "_v_400.jpg", vel_min, vel_max)
        plot_grid(u_400[0], test_case.name, test_case.name + "_u_400.jpg", vel_min, vel_max)

        for i in range(0, int(200.0 * (0.1 / DT))):
            test_case.step()
        test_case.save_gradients("_600")

        v_600 = test_case.get_velocity_y()
        u_600 = test_case.get_velocity_x()
        den_600 = test_case.get_density()
        plot_grid(den_600[0], test_case.name, test_case.name + "_den_600.jpg", den_min, den_max)
        plot_grid(v_600[0], test_case.name, test_case.name + "_v_600.jpg", vel_min, vel_max)
        plot_grid(u_600[0], test_case.name, test_case.name + "_u_600.jpg", vel_min, vel_max)

        for i in range(0, int(200.0 * (0.1 / DT))):
            test_case.step()
        test_case.save_gradients("_800")
        v_800 = test_case.get_velocity_y()
        u_800 = test_case.get_velocity_x()
        den_800 = test_case.get_density()
        plot_grid(den_800[0], test_case.name, test_case.name + "_den_800.jpg", den_min, den_max)
        plot_grid(v_800[0], test_case.name, test_case.name + "_v_800.jpg", vel_min, vel_max)
        plot_grid(u_800[0], test_case.name, test_case.name + "_u_800.jpg", vel_min, vel_max)

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
    case_1 = TestCase("Sin_xy", velocity_array, density_array, DT, vel_constant=True)
else:
    case_1 = TestCase("Sin_xy", velocity_field, density_field, DT, vel_constant=True)
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
    case_2 = TestCase("Sin_xy_2", velocity_array, density_array, DT)
else:
    case_2 = TestCase("Sin_xy_2", velocity_field, density_field, DT)
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
    case_3 = TestCase("Escape_1", velocity_array, density_array, DT, vel_constant=True)
else:
    case_3 = TestCase("Escape_1", velocity_field, density_field, DT, vel_constant=True)
TEST_CASES.append(case_3)


### Case 4 ###
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
    case_4 = TestCase("Escape_2", velocity_array, density_array, DT)
else:
    case_4 = TestCase("Escape_2", velocity_field, density_field, DT)
TEST_CASES.append(case_4)


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
    case_5 = TestCase("Simple_Stream", velocity_array, density_array, DT)
else:
    case_5 = TestCase("Simple_Stream", velocity_field, density_field, DT)
TEST_CASES.append(case_5)


### Case 6 ###
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
        next.append([-0.005 * (y - 50), -0.005 * (x - 50)])
    data.append(next)
velocity_array = np.array([data], dtype="float32")
velocity_field = StaggeredGrid(velocity_array)

if not semi_langrange_mode:
    case_6 = TestCase("Center_1", velocity_array, density_array, DT)
else:
    case_6 = TestCase("Center_1", velocity_field, density_field, DT)
TEST_CASES.append(case_6)



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
        if(x >= 40 and x < 60 and y >= 40 and y < 60):
            next.append([0.1, 0.2])
        else:
            next.append([0.1, 0.1])
    data.append(next)
velocity_array = np.array([data], dtype="float32")
velocity_field = StaggeredGrid(velocity_array)

if not semi_langrange_mode:
    case_7a = TestCase("Vel_X_Block", velocity_array, density_array, DT)
else:
    case_7a = TestCase("Vel_X_Block", velocity_field, density_field, DT)
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
        if(x >= 40 and x < 60 and y >= 40 and y < 60):
            next.append([-0.1, 0.2])
        else:
            next.append([-0.1, 0.1])
    data.append(next)
velocity_array = np.array([data], dtype="float32")
velocity_field = StaggeredGrid(velocity_array)

if not semi_langrange_mode:
    case_7b = TestCase("Vel_X_Block_2", velocity_array, density_array, DT)
else:
    case_7b = TestCase("Vel_X_Block_2", velocity_field, density_field, DT)
TEST_CASES.append(case_7b)


### Case 7c ###
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
        if(x >= 40 and x < 60 and y >= 40 and y < 60):
            next.append([-0.1, -0.2])
        else:
            next.append([-0.1, -0.1])
    data.append(next)
velocity_array = np.array([data], dtype="float32")
velocity_field = StaggeredGrid(velocity_array)

if not semi_langrange_mode:
    case_7c = TestCase("Vel_X_Block_3", velocity_array, density_array, DT)
else:
    case_7c = TestCase("Vel_X_Block_3", velocity_field, density_field, DT)
TEST_CASES.append(case_7c)


### Case 7d ###
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
        if(x >= 40 and x < 60 and y >= 40 and y < 60):
            next.append([0.1, -0.2])
        else:
            next.append([0.1, -0.1])
    data.append(next)
velocity_array = np.array([data], dtype="float32")
velocity_field = StaggeredGrid(velocity_array)

if not semi_langrange_mode:
    case_7d = TestCase("Vel_X_Block_4", velocity_array, density_array, DT)
else:
    case_7d = TestCase("Vel_X_Block_4", velocity_field, density_field, DT)
TEST_CASES.append(case_7d)



### Case 8a ###
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
        if(x >= 40 and x < 60 and y >= 40 and y < 60):
            next.append([0.2, 0.1])
        else:
            next.append([0.1, 0.1])
    data.append(next)
velocity_array = np.array([data], dtype="float32")
velocity_field = StaggeredGrid(velocity_array)

if not semi_langrange_mode:
    case_8a = TestCase("Vel_Y_Block", velocity_array, density_array, DT)
else:
    case_8a = TestCase("Vel_Y_Block", velocity_field, density_field, DT)
TEST_CASES.append(case_8a)


### Case 8b ###
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
        if(x >= 40 and x < 60 and y >= 40 and y < 60):
            next.append([0.2, -0.1])
        else:
            next.append([0.1, -0.1])
    data.append(next)
velocity_array = np.array([data], dtype="float32")
velocity_field = StaggeredGrid(velocity_array)

if not semi_langrange_mode:
    case_8b = TestCase("Vel_Y_Block_2", velocity_array, density_array, DT)
else:
    case_8b = TestCase("Vel_Y_Block_2", velocity_field, density_field, DT)
TEST_CASES.append(case_8b)


### Case 8c ###
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
        if(x >= 40 and x < 60 and y >= 40 and y < 60):
            next.append([-0.2, -0.1])
        else:
            next.append([-0.1, -0.1])
    data.append(next)
velocity_array = np.array([data], dtype="float32")
velocity_field = StaggeredGrid(velocity_array)

if not semi_langrange_mode:
    case_8c = TestCase("Vel_Y_Block_3", velocity_array, density_array, DT)
else:
    case_8c = TestCase("Vel_Y_Block_3", velocity_field, density_field, DT)
TEST_CASES.append(case_8c)


### Case 8d ###
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
        if(x >= 40 and x < 60 and y >= 40 and y < 60):
            next.append([-0.2, 0.1])
        else:
            next.append([-0.1, 0.1])
    data.append(next)
velocity_array = np.array([data], dtype="float32")
velocity_field = StaggeredGrid(velocity_array)

if not semi_langrange_mode:
    case_8d = TestCase("Vel_Y_Block_4", velocity_array, density_array, DT)
else:
    case_8d = TestCase("Vel_Y_Block_4", velocity_field, density_field, DT)
TEST_CASES.append(case_8d)



### Case 9 ###
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
        if(x >= 40 and x < 60 and y >= 40 and y < 60):
            next.append([-0.2, 0.0])
        else:
            next.append([0.0, 0.0])
    data.append(next)
velocity_array = np.array([data], dtype="float32")
velocity_field = StaggeredGrid(velocity_array)

if not semi_langrange_mode:
    case_9 = TestCase("Vel_Y_Only", velocity_array, density_array, DT)
else:
    case_9 = TestCase("Vel_Y_Only", velocity_field, density_field, DT)
TEST_CASES.append(case_9)


### Case 10 ###
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
        if(x >= 40 and x < 60 and y >= 40 and y < 60):
            next.append([0.0, -0.2])
        else:
            next.append([0.0, 0.0])
    data.append(next)
velocity_array = np.array([data], dtype="float32")
velocity_field = StaggeredGrid(velocity_array)

if not semi_langrange_mode:
    case_10 = TestCase("Vel_X_Only", velocity_array, density_array, DT)
else:
    case_10 = TestCase("Vel_X_Only", velocity_field, density_field, DT)
TEST_CASES.append(case_10)


### Case 11 ###
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
        if(x >= 40 and x < 60 and y >= 40 and y < 60):
            next.append([0.1, -0.1])
        else:
            next.append([0.0, 0.0])
    data.append(next)
velocity_array = np.array([data], dtype="float32")
velocity_field = StaggeredGrid(velocity_array)

if not semi_langrange_mode:
    case_11 = TestCase("Vel_XY", velocity_array, density_array, DT)
else:
    case_11 = TestCase("Vel_XY", velocity_field, density_field, DT)
TEST_CASES.append(case_11)


### CASE 12 - Breaking Time ###
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
        u = 0.2 * EXP0 ** -((2*(0.03125*x-1))**2)
        next.append([0.0, u])
    data.append(next)
velocity_array = np.array([data], dtype="float32")
velocity_field = StaggeredGrid(velocity_array)
if not semi_langrange_mode:
    case_12 = TestCase("Burgers_1", velocity_array, density_array, DT)
else:
    case_12 = TestCase("Burgers_1", velocity_field, density_field, DT)
TEST_CASES.append(case_12)



### CASE 13 - Breaking Time 2 ###
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
        if(x <= 20):
            next.append([0.0, 0.2])
        else:
            next.append([0.0, 0.0])
    data.append(next)
velocity_array = np.array([data], dtype="float32")
velocity_field = StaggeredGrid(velocity_array)
if not semi_langrange_mode:
    case_13 = TestCase("Discont_X", velocity_array, density_array, DT)
else:
    case_13 = TestCase("Discont_X", velocity_field, density_field, DT)
TEST_CASES.append(case_13)


### CASE 14 - Breaking Time ###
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
        u = 0.2 * EXP0 ** -((2*(0.0625*x-1))**2) + 0.1 * EXP0 ** -((2*(0.0625*(x-18)-1))**2)
        next.append([0.0, u])
    data.append(next)
velocity_array = np.array([data], dtype="float32")
velocity_field = StaggeredGrid(velocity_array)
if not semi_langrange_mode:
    case_14 = TestCase("WaveChase", velocity_array, density_array, DT)
else:
    case_14 = TestCase("WaveChase", velocity_field, density_field, DT)
TEST_CASES.append(case_14)




#cProfile.run('run_test_cases(TEST_CASES)')
run_test_cases(TEST_CASES)
