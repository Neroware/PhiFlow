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
prev_step_derivates = {}

def quick_advection(field, velocity_field, dt, type="density"):
    def quick_advection_density(density_field, velodity_field, dt):
        # Neighboring grid points have a distance of 1.0
        delta_x = delta_y = 1.0

        # Sample points (i,j)
        points = field.points
        # Density values at sample points
        density_data = density_field.sample_at(points.data)[0]
        # Grid Dimensions in Sample Points
        dim_y = len(density_data)
        dim_x = len(density_data[0])
        #print("Dimension in Sample Points: ", dim_x, ", ", dim_y)

        # Get velocity values
        # u at (i+0.5,j), v at (i,j+0.5)
        vel_v, vel_u = velocity_field.unstack()
        #print("Staggered grid velocity u: ", vel_u.data)
        #print("Staggered grid velocity v: ", vel_v.data)
        
        # Interpolate density at (i+0.5,j) and (i,j+0.5)
        staggered_density_x = []
        staggered_density_y = []
        for j in range(0, dim_y):
            row = []
            for i in range(0, dim_x + 1):
                value1 = value2 = 0.0
                if(i == 0):
                    value2 = density_data[j][i][0]
                elif(i == dim_x):
                    value1 = density_data[j][i - 1][0]
                else:
                    value1 = density_data[j][i - 1][0]
                    value2 = density_data[j][i][0]
                rho = 0.5 * (value1 + value2)
                row.append(rho)
            staggered_density_x.append(row)
        for j in range(0, dim_y + 1):
            row = []
            for i in range(0, dim_x):
                value1 = value2 = 0.0
                if(j == 0):
                    value2 = density_data[j][i][0]
                elif(j == dim_y):
                    value1 = density_data[j - 1][i][0]
                else:
                    value1 = density_data[j - 1][i][0]
                    value2 = density_data[j][i][0]
                rho = 0.5 * (value1 + value2)
                row.append(rho)
            staggered_density_y.append(row)

        print("Density Data:\n", density_data)
        print("Staggered Density Grid X: ", staggered_density_x)
        print("Staggered Density Grid Y: ", staggered_density_y)
        print("=====================================================\n")

        # Calculate partial derviates dup/dx, dvp/dy and finally dp/dt with velocity (u, v) and density p
        derivates_grid = np.zeros((dim_y, dim_x))
        density_data_step = density_data.copy()
        vel_u_data = vel_u.data[0]
        vel_v_data = vel_v.data[0]

        for j in range(0, dim_y):
            for i in range(0, dim_x):
                # Discretize partial derivates
                rho_x1 = staggered_density_x[j][i]
                rho_x2 = staggered_density_x[j][i + 1]
                if(i == 0 and j == 1):
                    print("Density at (0.5, 1) and (1.5, 1): ", rho_x1, ", ", rho_x2)
                u1 = vel_u_data[j][i][0]
                u2 = vel_u_data[j][i + 1][0]
                dupdx = (u2 * rho_x2 - u1 * rho_x1) / delta_x
                if(i == 0 and j == 1):
                    print("Density partial derivate at (1, 1): ", dupdx)

                rho_y1 = staggered_density_y[j][i]
                rho_y2 = staggered_density_y[j + 1][i]
                v1 = vel_v_data[j][i][0]
                v2 = vel_v_data[j + 1][i][0]
                dvpdy = (v2 * rho_y2 - v1 * rho_y1) / delta_y

                # Solve Advection Equation
                dpdt = -dupdx - dvpdy
                derivates_grid[j][i] = dpdt

                # Perform Explicit Euler Step
                density_data_step[j][i][0] += derivates_grid[j][i] * dt

        return CenteredGrid([density_data_step])
        #raise NotImplementedError("QUICK for Density is WIP!")


    def quick_advection_velocity(velocity_field, dt):
        raise NotImplementedError("QUICK for Velocity is WIP!")


    if(type == "density"):
        return quick_advection_density(field, velocity_field, dt)
    elif(type == "velocity"):
        return quick_advection_velocity(velocity_field, dt)

    

### =========================== End QUICK =========================== ###




class SimpleFlow(App):

    def __init__(self):
        App.__init__(self, 'Small Flow', DESCRIPTION, summary='fluid' + 'x'.join([str(d) for d in RESOLUTION]), framerate=20)

        self.physics = SimpleFlowPhysics()
        #self.physics = IncompressibleFlow()
        self.timestep = 1.0

        fluid = self.fluid = world.add(Fluid(Domain(RESOLUTION, box=box[0:3, 0:3], boundaries=OPEN), buoyancy_factor=0.0), physics=self.physics)
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
        self.fluid.density = quick_advection(density, velocity, dt, type="density")
        #self.fluid.velocity = quick_advection(velocity, velocity, dt, type="velocity")

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
                if x % 3 == 0 and y % 3 == 0:
                    next.append([1.0])
                elif x % 3 > 0 and y % 3 == 0:
                    next.append([2.0])
                elif x % 3 == 0 and y % 3 > 0:
                    next.append([2.0])
                else:
                    next.append([1.0])
            data.append(next)

        density_array = np.array([data])
        return CenteredGrid(density_array)


    def _get_velocity_grid(self):
        """
        Generiert a StaggeredGrid, with constant velocity (u, v) := (1, 0)
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