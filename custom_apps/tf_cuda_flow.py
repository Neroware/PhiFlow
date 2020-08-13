import sys
if not 'tf' in sys.argv:
    raise RuntimeError("This simulation can only be run in TensorFlow-Mode!")
from phi.tf.flow import *  # Use TensorFlow
MODE = 'TensorFlow'
RESOLUTION = [int(sys.argv[1])] * 2 if len(sys.argv) > 1 and __name__ == '__main__' else [128] * 2
DESCRIPTION = "Basic fluid test that runs QUICK Scheme with CUDA"

from phi.tf.app import App
from phi.tf.tf_cuda_quick_advection import *

import math
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


class CUDAFlow(App):
    def __init__(self):
        App.__init__(self, 'CUDA Flow', DESCRIPTION, summary='fluid' + 'x'.join([str(d) for d in RESOLUTION]), framerate=20) 

        self.physics = SimpleFlowPhysics()
        #self.physics = SemiLangFlowPhysics()
        self.timestep = 0.1

        fluid = self.fluid = world.add(Fluid(Domain(RESOLUTION, box=box[0:100, 0:100], boundaries=OPEN), buoyancy_factor=0.0), physics=self.physics)
        fluid.velocity = self._get_velocity_grid()
        fluid.density = self._get_density_grid_2()
        #fluid.density = self._get_density_grid()
        #world.add(ConstantVelocity(box[0:100, 0:100], velocity=(1, 0)))

        self.add_field('Velocity', lambda: fluid.velocity)
        self.add_field('Density', lambda: fluid.density)


    def step(self):
        velocity = self.fluid.velocity
        density = self.fluid.density
        dt = self.timestep
        dim = RESOLUTION[0]

        #grds_x, grds_y = tf_quick_density_gradients(density, velocity, 1, 1, dt)
        #print("Gradient: ", grds_x)
        #grds = tf_semi_lagrange_density_gradients(density, velocity, dt)
        #with tf.Session("") as sess:
        #    for grd in grds_y:
        #        print(">>> Grd.: ", grd.eval())
        #    sess.close()

        # This is ugly but I but since this is siumlation code it's not too bad
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
            self.fluid.density = CenteredGrid(den.eval())
            self.fluid.velocity = to_staggered_grid(vel_u.eval()[0], vel_v.eval()[0], dim)
            sess.close()

        world.step(dt=self.timestep)
        

    def action_reset(self):
        self.steps = 0
        self.fluid.density = self.fluid.velocity = 0


    def _get_density_grid(self):
        """
        Generates a chessboard-like density grid
        """
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
        return CenteredGrid(density_array)


    def _get_density_grid_2(self):
        """
        Generates a single square in the center of the grid
        """
        data = []
        for y in range(0, RESOLUTION[0]):
            next = []
            for x in range(0, RESOLUTION[0]):
                #if(x >= 45 and x <= 55 and y >= 45 and y <= 55):
                #    next.append([0.2])
                #else:
                #    next.append([0.0])
                next.append([0.1])
            data.append(next)

        density_array = np.array([data], dtype="float32")
        return CenteredGrid(density_array)


    def _get_density_grid_3(self):
        """
        Generates a single spot in the center of the grid
        """
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
                next.append([0.1 * m])
            data.append(next)

        density_array = np.array([data], dtype="float32")
        return CenteredGrid(density_array)


    def _get_velocity_grid(self):
        """
        Generates a StaggeredGrid, with constant velocity u = (0, 1)
        """

        data = []
        for y in range(0, RESOLUTION[0] + 1):
            next = []
            for x in range(0, RESOLUTION[0] + 1):
                #if(y >= 45 and y <= 55 and x >= 45 and x <= 55):
                #    next.append([0.1, 0.2])
                #else:
                #    next.append([0.1, 0.1])

                #next.append([-0.2, 0.3])
                
                if(x == 50):
                    next.append([-0.2, 0.1])
                else:
                    next.append([0.0, 0.1])
                #next.append([0.0, 0.1])

                #next.append([0.02 * (y - 50), 0.02 * (x - 50)])

                #if(x == 1):
                #    next.append([0.1, 0.2])
                #else:
                #    next.append([0.1, 0.1])

                #next.append([0.1 * math.sin(0.02 * PI * y), 0.1 * math.sin(0.02 * PI * x)])
                
                #next.append([-0.5, -0.2 * (y / 100.0)])

                #vx = x - 50
                #vy = y - 50
                #if(vx == 0.0 and vy == 0.0):
                #    next.append([0.0, 0.0])
                #else:
                #    m = 1.0 / math.sqrt(vx * vx + vy * vy)
                #    vx = 0.5 * m * vx
                #    vy = 0.5 * m * vy
                #    next.append([vy, vx])
            data.append(next)

        velocity_grid = np.array([data], dtype="float32")
        return StaggeredGrid(velocity_grid)


show(CUDAFlow(), display=('Velocity', 'Density'), framerate=2)
