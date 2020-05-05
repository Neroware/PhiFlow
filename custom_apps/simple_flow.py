import numpy as np

import sys
if 'tf' in sys.argv:
    from phi.tf.flow import *  # Use TensorFlow
    MODE = 'TensorFlow'
else:
    from phi.flow import *  # Use NumPy
    MODE = 'NumPy'
RESOLUTION = [int(sys.argv[1])] * 2 if len(sys.argv) > 1 and __name__ == '__main__' else [128] * 2
DESCRIPTION = "Very basic flow test."


class SimpleFlow(App):

    def __init__(self):
        App.__init__(self, 'Simple Flow', DESCRIPTION, summary='fluid' + 'x'.join([str(d) for d in RESOLUTION]), framerate=20)
        fluid = self.fluid = world.add(Fluid(Domain(RESOLUTION, box=box[0:100, 0:100], boundaries=OPEN), buoyancy_factor=0.1), physics=IncompressibleFlow())

        fluid.velocity = self._get_velocity_grid()
        fluid.density = self._get_density_grid()
        #world.add(ConstantVelocity(box[0:100, 0:100], velocity=(1, 0)))

        self.add_field('Velocity', lambda: fluid.velocity)
        self.add_field('Density', lambda: fluid.density)


    def step(self):
        world.step()
        

    def action_reset(self):
        self.steps = 0
        self.fluid.density = self.fluid.velocity = 0


    def _get_density_grid(self):
        """
        Generiert ein Schachbrettmuster als CenteredGrid
        """
        data = []
        for x in range(0, RESOLUTION[0]):
            next = []
            for y in range(0, RESOLUTION[0]):
                if x % 8 <= 3 and y % 8 <= 3:
                    next.append([0.5])
                elif x % 8 > 3 and y % 8 <= 3:
                    next.append([1.0])
                elif x % 8 <= 3 and y % 8 > 3:
                    next.append([1.0])
                else:
                    next.append([0.5])
            data.append(next)

        density_array = np.array([data])
        return CenteredGrid(density_array)


    def _get_velocity_grid(self):
        """
        Generiert ein StaggeredGrid, welches eine konstante Geschwindigkeit mit u = (1, 0) repraesentiert
        """
        data = []
        for x in range(0, RESOLUTION[0] + 1):
            next = []
            for y in range(0, RESOLUTION[0] + 1):
                next.append([1.0, 0.0])
            data.append(next)

        velocity_tensor = np.array([data])
        return StaggeredGrid(velocity_tensor)




show(SimpleFlow(), display=('Velocity', 'Desnity'), framerate=2)