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




### =========================== QUICK Advection Algorithm =========================== ###
#import numpy as np
#import tensorflow as tf
prev_step_derivates = {}

def quick_advection(field, velocity_field, dt, type_id="default"):
    try:
        points = field.points
        # Field values
        values = field.sample_at(points.data)[0]
        # Neighboring grid points are having a distance of 1.0
        deltaX = deltaY = 1.0

        # X- and Y-components for velocity = (u, v)
        # Position: (i+0.5,j)
        vel = velocity_field.sample_at(points.data)[0]
        vel_v = np.array(list(map(lambda row: np.array(list(map(lambda pos: [pos[0]], row))), vel)))
        vel_u = np.array(list(map(lambda row: np.array(list(map(lambda pos: [pos[1]], row))), vel)))
        # Position: (i+0.5,j+1)
        points2 = CenteredGrid( np.array(list(map(lambda row: np.array(list(map(lambda pos: pos + [1.0, 0.0], row))), points.data))) )
        vel2 = velocity_field.sample_at(points2.data)[0]
        vel2_u = np.array(list(map(lambda row: np.array(list(map(lambda pos: [pos[1]], row))), vel2)))
        # u at Position: (i+0.5,j+0.5) # Equation (3)
        vel3_u = 0.5 * (vel_u + vel2_u)
        # Position: (i-1,j+0.5)
        points4 = CenteredGrid( np.array(list(map(lambda row: np.array(list(map(lambda pos: pos + [0.5, -1.5], row))), points.data))) )
        vel4 = velocity_field.sample_at(points4.data)[0]
        vel4_v = np.array(list(map(lambda row: np.array(list(map(lambda pos: [pos[0]], row))), vel4)))
        # Position: (i,j+0.5)
        points5 = CenteredGrid( np.array(list(map(lambda row: np.array(list(map(lambda pos: pos + [0.5, -0.5], row))), points.data))) )
        vel5 = velocity_field.sample_at(points5.data)[0]
        vel5_v = np.array(list(map(lambda row: np.array(list(map(lambda pos: [pos[0]], row))), vel5)))
        # Position: (i+1,j+0.5)
        points6 = CenteredGrid( np.array(list(map(lambda row: np.array(list(map(lambda pos: pos + [0.5, 0.5], row))), points.data))) )
        vel6 = velocity_field.sample_at(points6.data)[0]
        vel6_v = np.array(list(map(lambda row: np.array(list(map(lambda pos: [pos[0]], row))), vel6)))
        # v at Position: (i+0.5,j+0.5) # Equation (4)
        vel3_v = vel6_v.copy()
        for j in range(0, len(vel3_u)):
            for i in range(0, len(vel3_u[0])):
                u = vel3_u[j][i][0]
                v1 = vel4_v[j][i][0]
                v2 = vel5_v[j][i][0]
                v3 = vel6_v[j][i][0]
                if(u >= 0):
                    v = [0.125 * v1 + 0.25 * v2 + 0.625 * v3]
                    vel3_v[j][i] = v
                else:
                    v = [0.625 * v1 + 0.25 * v2 + 0.125 * v3]
                    vel3_v[j][i] = v

        # Generate partial derivate field # Equation (2)
        # In X-Direction
        derivate_field_x = values.copy()
        derivate_field_y = values.copy()
        derivate_field_t = values.copy()
        # Get value data at (i+0.5,j+0.5) and (i-0.5,j+0.5)
        value_points1 = CenteredGrid( np.array(list(map(lambda row: np.array(list(map(lambda pos: pos + [0.5, 0.0], row))), points.data))) )
        value_points2 = CenteredGrid( np.array(list(map(lambda row: np.array(list(map(lambda pos: pos + [0.5, -1.0], row))), points.data))) )
        values1 = field.sample_at(value_points1.data)[0]
        values2 = field.sample_at(value_points2.data)[0]

        # We need to solve dU/dt = -d(uU)/dx - d(vU)/dy
        for j in range(0, len(values)):
            for i in range(0, len(values[0])):
                u1 = vel3_u[j][i][0]
                v1 = vel3_v[j][i][0]
                u2 = u1
                v2 = v1
                if(i > 0):
                    u2 = vel3_u[j][i-1][0]
                    v2 = vel3_v[j][i-1][0]
                
                value1 = values1[j][i][0]
                value2 = values2[j][i][0]

                duUdx = (u1 * value1 - u2 * value2) / deltaX
                dvUdy = (v1 * value1 - v2 * value2) / deltaY
                derivate_field_x[j][i] = np.array([duUdx])
                derivate_field_y[j][i] = np.array([dvUdy])
                derivate_field_t[j][i] = -1.0 * derivate_field_x[j][i] - derivate_field_y[j][i]

        # Perform a timestep. If there is no previous timestep saved we do Euler, otherwise we do Adams-Bashforth
        if(not type_id in prev_step_derivates):
            data = values.copy()
            for j in range(0, len(data)):
                for i in range(0, len(data[0])):
                    data[j][i] += derivate_field_t[j][i] * dt
            prev_step_derivates[type_id] = derivate_field_t
            return field.with_data(np.array([data]))
        else:
            raise NotImplementedError("Adams-Bashforth Timestep is WIP!")

    except StaggeredSamplePoints:
        unstacked = field.unstack()
        advected = []
        for index in range(0, len(unstacked)):
            component = unstacked[index]
            advected.append(quick_advection(component, velocity_field, dt, type_id + str(index)))
        return field.with_data(advected)

### =========================== End QUICK =========================== ###




class SimpleFlow(App):

    def __init__(self):
        App.__init__(self, 'Simple Flow', DESCRIPTION, summary='fluid' + 'x'.join([str(d) for d in RESOLUTION]), framerate=20)

        self.physics = SimpleFlowPhysics()
        #self.physics = IncompressibleFlow()
        self.timestep = 1.0

        fluid = self.fluid = world.add(Fluid(Domain(RESOLUTION, box=box[0:100, 0:100], boundaries=OPEN), buoyancy_factor=0.0), physics=self.physics)
        fluid.velocity = self._get_velocity_grid()
        fluid.density = self._get_density_grid()
        #world.add(ConstantVelocity(box[0:100, 0:100], velocity=(1, 0)))

        self.add_field('Velocity', lambda: fluid.velocity)
        self.add_field('Density', lambda: fluid.density)


    def step(self):
        velocity = self.fluid.velocity
        density = self.fluid.density
        dt = self.timestep

        # Advection
        self.fluid.density = quick_advection(density, velocity, dt, type_id="density")
        self.fluid.velocity = quick_advection(velocity, velocity, dt, type_id="velocity")

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

        density_array = np.array([data])
        return CenteredGrid(density_array)


    def _get_velocity_grid(self):
        """
        Generiert a StaggeredGrid, with constant velocity u = (0, 1)
        """
        data = []
        for y in range(0, RESOLUTION[0] + 1):
            next = []
            for x in range(0, RESOLUTION[0] + 1):
                next.append([0.0, 1.0])
            data.append(next)

        velocity_grid = np.array([data])
        return StaggeredGrid(velocity_grid)




show(SimpleFlow(), display=('Velocity', 'Desnity'), framerate=2)