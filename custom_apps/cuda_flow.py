import sys
if not 'tf' in sys.argv:
    raise RuntimeError("This simulation can only be run in TensorFlow-Mode!")
from phi.tf.flow import *  # Use TensorFlow
MODE = 'TensorFlow'
RESOLUTION = [int(sys.argv[1])] * 2 if len(sys.argv) > 1 and __name__ == '__main__' else [128] * 2
DESCRIPTION = "Basic fluid test that runs QUICK Scheme with CUDA"

from phi.tf.tf_cuda_quick_advection import tf_cuda_quick_advection



class CUDAFlow(App):
    def __init__(self):
        App.__init__(self, 'CUDA Flow', DESCRIPTION, summary='fluid' + 'x'.join([str(d) for d in RESOLUTION]), framerate=20) 

        self.physics = SimpleFlowPhysics()
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

        #self.fluid.density = tf_cuda_quick_advection(velocity, dt, field=density, field_type="density")
        self.fluid.velocity = tf_cuda_quick_advection(velocity, dt, field_type="velocity")

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
        Generates a single spot in the center of the grid
        """
        data = []
        for y in range(0, RESOLUTION[0]):
            next = []
            for x in range(0, RESOLUTION[0]):
                if(x >= 45 and x <= 55 and y >= 45 and y <= 55):
                    next.append([1.0])
                else:
                    next.append([0.0])
            data.append(next)

        density_array = np.array([data], dtype="float32")
        return CenteredGrid(density_array)


    def _get_velocity_grid(self):
        """
        Generates a StaggeredGrid, with constant velocity u = (0, 1)
        """
        #print("Generating Demo StaggeredGrid...")
        #t_data = np.array([[[[0.0, 1.0], [2.0, 3.0], [None, 5.0]], [[6.0, 7.0], [8.0, 9.0], [None, 11.0]], [[12.0, None], [14.0, None], [None, None]]]], dtype="float32")
        #t_grid = StaggeredGrid(t_data)
        #t_data_1, t_data_2 = t_grid.data
        #print("---> ", t_data_1.data)
        #print(">>>> ", t_data_2.data)
        #print("Test End!")

        data = []
        for y in range(0, RESOLUTION[0] + 1):
            next = []
            for x in range(0, RESOLUTION[0] + 1):
                #next.append([0.0, 1.0])
                if(y == 25):
                    next.append([1.1, 0.0])
                else:
                    next.append([1.0, 0.0])
            data.append(next)

        velocity_grid = np.array([data], dtype="float32")
        return StaggeredGrid(velocity_grid)


show(CUDAFlow(), display=('Velocity', 'Density'), framerate=2)
