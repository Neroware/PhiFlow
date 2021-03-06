import sys
if not 'tf' in sys.argv:
    raise RuntimeError("This simulation can only be run in TensorFlow-Mode!")
from phi.tf.flow import *  # Use TensorFlow
MODE = 'TensorFlow'
RESOLUTION = [8] * 2 #[int(sys.argv[1])] * 2 if len(sys.argv) > 1 and __name__ == '__main__' else [128] * 2
DESCRIPTION = "Basic fluid test that runs QUICK Scheme with CUDA"

from phi.tf.app import App
from phi.tf.tf_cuda_quick_advection import *
from phi.physics.field.advect import *

import math
import matplotlib.pyplot as plt
from matplotlib import cm

PI = 3.14159


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


def plot_grid(data, path, min_value, max_value):
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
    fig.savefig(path)
    plt.close()
 


class CUDAFlow(App):
    def __init__(self):
        App.__init__(self, 'CUDA Flow', DESCRIPTION, summary='fluid' + 'x'.join([str(d) for d in RESOLUTION]), framerate=20) 
        self.physics = SimpleFlowPhysics()
        self.timestep = 0.1
        fluid = self.fluid = world.add(Fluid(Domain(RESOLUTION, box=box[0:8, 0:8], boundaries=OPEN), buoyancy_factor=0.0), physics=self.physics)
        fluid.velocity = self._get_velocity_grid()
        fluid.density = self._get_density_grid()
        self.add_field('Velocity', lambda: fluid.velocity)
        self.add_field('Density', lambda: fluid.density)


    def step(self):
        velocity = self.fluid.velocity
        density = self.fluid.density
        dt = self.timestep
        dim = RESOLUTION[0]

        # Machine Learning with Gradient Descent
        self._gradient_descent_quick(density, velocity, dt, dim)
        self._gradient_descent_semi_lagrange(density, velocity, dt)

        # Init tensors
        density_tensor = tf.constant(density.data)
        density_tensor_padded = tf.constant(density.padded(2).data)
        velocity_v_field, velocity_u_field = velocity.data
        velocity_v_tensor = tf.constant(velocity_v_field.data)
        velocity_u_tensor = tf.constant(velocity_u_field.data)
        velocity_v_tensor_padded = tf.constant(velocity_v_field.padded(2).data)
        velocity_u_tensor_padded = tf.constant(velocity_u_field.padded(2).data)

        # Perform QUICK Advection Step
        den = tf_cuda_quick_advection(density_tensor, density_tensor_padded, velocity_u_tensor_padded, velocity_v_tensor_padded, dt, dim, dim, 1.0, 1.0, field_type="density")
        vel_u = tf_cuda_quick_advection(velocity_u_tensor, velocity_u_tensor_padded, velocity_u_tensor_padded, velocity_v_tensor_padded, dt, dim, dim, 1.0, 1.0, field_type="velocity_u")
        vel_v = tf_cuda_quick_advection(velocity_v_tensor, velocity_v_tensor_padded, velocity_u_tensor_padded, velocity_v_tensor_padded, dt, dim, dim, 1.0, 1.0, field_type="velocity_v")
        with tf.Session("") as sess:
            self.fluid.density = CenteredGrid(den.eval())
            self.fluid.velocity = to_staggered_grid(vel_u.eval()[0], vel_v.eval()[0], dim)
            sess.close()
        world.step(dt=self.timestep)
        

    def action_reset(self):
        self.steps = 0
        self.fluid.density = self.fluid.velocity = 0


    def _gradient_descent_semi_lagrange(self, density, velocity, dt):
        density_tensor = tf.Variable(density.data)
        density_field = CenteredGrid(density_tensor)
        velocity_v_field, velocity_u_field = velocity.data
        velocity_v_tensor = tf.Variable(velocity_v_field.data)
        velocity_u_tensor = tf.Variable(velocity_u_field.data)
        target_field = tf.constant(self._get_target_field().data)
        velocity_field = StaggeredGrid((velocity_v_tensor, velocity_u_tensor))
        plot_grid(density.data[0], "tf_cuda_grad/sl/tf_cuda_init_rho.jpg", -0.4, 0.4)
        plot_grid(velocity_v_field.data[0], "tf_cuda_grad/sl/tf_cuda_init_v.jpg", -0.4, 0.4)
        plot_grid(velocity_u_field.data[0], "tf_cuda_grad/sl/tf_cuda_init_u.jpg", -0.4, 0.4)
        plot_grid(self._get_target_field().data[0], "tf_cuda_grad/sl/tf_cuda_target.jpg", -0.4, 0.4)

        rho_adv = semi_lagrangian(density_field, velocity_field, dt).data
        L = target_field - rho_adv
        optimizer = tf.train.GradientDescentOptimizer(0.1)
        train = optimizer.minimize(L)
        print("Got training results:\n ", train)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run(train)
        result = sess.run((density_tensor, velocity_u_tensor, velocity_v_tensor, L))
        plot_grid(result[0][0], "tf_cuda_grad/sl/tf_cuda_grad_rho.jpg", -0.4, 0.4)
        plot_grid(result[1][0], "tf_cuda_grad/sl/tf_cuda_grad_u.jpg", -0.4, 0.4)
        plot_grid(result[2][0], "tf_cuda_grad/sl/tf_cuda_grad_v.jpg", -0.4, 0.4)
        plot_grid(result[3][0], "tf_cuda_grad/sl/tf_cuda_grad_Loss.jpg", -0.4, 0.4)
        print(result)
        print("=====================================")


    def _gradient_descent_quick(self, density, velocity, dt, dim):
        density_tensor = tf.Variable(density.data)
        density_tensor_padded = tf.Variable(density.padded(2).data)
        velocity_v_field, velocity_u_field = velocity.data
        velocity_v_tensor = tf.Variable(velocity_v_field.data)
        velocity_u_tensor = tf.Variable(velocity_u_field.data)
        velocity_v_tensor_padded = tf.Variable(velocity_v_field.padded(2).data)
        velocity_u_tensor_padded = tf.Variable(velocity_u_field.padded(2).data)
        target = tf.constant(self._get_target_field().data)
        plot_grid(density.data[0], "tf_cuda_grad/quick/tf_cuda_init_rho.jpg", -0.4, 0.4)
        plot_grid(velocity_v_field.data[0], "tf_cuda_grad/quick/tf_cuda_init_v.jpg", -0.4, 0.4)
        plot_grid(velocity_u_field.data[0], "tf_cuda_grad/quick/tf_cuda_init_u.jpg", -0.4, 0.4)
        plot_grid(self._get_target_field().data[0], "tf_cuda_grad/quick/tf_cuda_target.jpg", -0.4, 0.4)

        rho_adv = tf_cuda_quick_advection(density_tensor, density_tensor_padded, velocity_u_tensor_padded, velocity_v_tensor_padded, dt, dim, dim, 1.0, 1.0, field_type="density")
        L = target - rho_adv
        optimizer = tf.train.GradientDescentOptimizer(0.1)
        train = optimizer.minimize(L)
        print("Got training results:\n ", train)
        
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run(train)
        result = sess.run((density_tensor_padded, velocity_u_tensor_padded, velocity_v_tensor_padded, L))
        plot_grid(result[0][0], "tf_cuda_grad/quick/tf_cuda_grad_rho.jpg", -0.4, 0.4)
        plot_grid(result[1][0], "tf_cuda_grad/quick/tf_cuda_grad_u.jpg", -0.4, 0.4)
        plot_grid(result[2][0], "tf_cuda_grad/quick/tf_cuda_grad_v.jpg", -0.4, 0.4)
        plot_grid(result[3][0], "tf_cuda_grad/quick/tf_cuda_grad_Loss.jpg", -0.4, 0.4)
        print(result)
        print("=====================================")


    def _get_velocity_grid(self):
        """
        Generates a StaggeredGrid, with constant velocity u = (0, 1)
        """
        data = []
        for y in range(0, RESOLUTION[0] + 1):
            next = []
            for x in range(0, RESOLUTION[0] + 1):
                dim = RESOLUTION[0]
                next.append([0.1, 0.1])
            data.append(next)
        velocity_grid = np.array([data], dtype="float32")
        return StaggeredGrid(velocity_grid)


    def _get_target_field(self):
        data = []
        for y in range(0, RESOLUTION[0]):
            next = []
            for x in range(0, RESOLUTION[0]):
                if((x == 2 or x == 3) and (y == 2 or y == 3)):
                    next.append([0.1])
                else:
                    next.append([0.0])
            data.append(next)
        density_array = np.array([data], dtype="float32")
        return CenteredGrid(density_array)


    def _get_density_grid(self):
        data = []
        for y in range(0, RESOLUTION[0]):
            next = []
            for x in range(0, RESOLUTION[0]):
                if((x == 3 or x == 4) and (y == 3 or y == 4)):
                    next.append([0.1])
                else:
                    next.append([0.0])
            data.append(next)
        density_array = np.array([data], dtype="float32")
        return CenteredGrid(density_array)


show(CUDAFlow(), display=('Velocity', 'Density'), framerate=2)
