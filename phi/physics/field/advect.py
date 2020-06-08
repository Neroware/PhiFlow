from phi.physics.field import SampledField, ConstantField, StaggeredGrid, CenteredGrid
from .field import StaggeredSamplePoints, Field


def advect(field, velocity, dt):
    """
Advect `field` along the `velocity` vectors using the default advection method.
    :param field: any built-in Field
    :type field: Field
    :param velocity: any Field
    :type velocity: Field
    :param dt: time increment
    :return: Advected field of same type as `field`
    """
    if isinstance(field, SampledField):
        return runge_kutta_4(field, velocity, dt=dt)
    if isinstance(field, ConstantField):
        return field
    if isinstance(field, (CenteredGrid, StaggeredGrid)):
        return semi_lagrangian(field, velocity, dt=dt)
    raise NotImplementedError(field)


def semi_lagrangian(field, velocity_field, dt):
    """
Semi-Lagrangian advection with simple backward lookup.
        :param field: Field to be advected
        :param velocity_field: Field, need not be compatible with field.
        :param dt: time increment
        :return: Field compatible with input field
    """
    try:
        x0 = field.points
        v = velocity_field.at(x0)
        x = x0 - v * dt
        data = field.sample_at(x.data)
        return field.with_data(data)
    except StaggeredSamplePoints:
        advected = [semi_lagrangian(component, velocity_field, dt) for component in field.unstack()]
        return field.with_data(advected)


def runge_kutta_4(field, velocity, dt):
    """
Lagrangian advection of particles.
    :param field: SampledField with any number of components
    :type field: SampledField
    :param velocity: Vector field
    :type velocity: Field
    :param dt: time increment
    :return: SampledField with same data as `field` but advected points
    """
    assert isinstance(field, SampledField)
    assert isinstance(velocity, Field)
    points = field.points
    # --- Sample velocity at intermediate points ---
    vel_k1 = velocity.at(points)
    vel_k2 = velocity.at(points + 0.5 * dt * vel_k1)
    vel_k3 = velocity.at(points + 0.5 * dt * vel_k2)
    vel_k4 = velocity.at(points + dt * vel_k3)
    # --- Combine points with RK4 scheme ---
    new_points = points + dt * (1/6.) * (vel_k1 + 2 * (vel_k2 + vel_k3) + vel_k4)
    result = SampledField(new_points.data, field.data, mode=field.mode, point_count=field._point_count, name=field.name)
    return result


def quick_advection(field, velocity_field, dt, type="density"):
    """
    QUICK Advection Scheme with Explicit Euler Step
    :param field: SampledField with any number of components
    :type field: SampledField
    :param velocity: Vector field
    :type velocity: Field
    :param dt: time increment
    :return: SampledField with same data as `field` but advected points
    
    (i) This function only works in Numpy-Mode, for TensorFlow a CUDA module will be used
    Too large timesteps cause instability!
    """
    import numpy as np

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

        # Get velocity values
        # u at (i+0.5,j), v at (i,j+0.5)
        vel_v, vel_u = velocity_field.unstack()
        vel_u_data = vel_u.data[0]
        vel_v_data = vel_v.data[0]
        
        # Upwind density at (i+0.5,j) and (i,j+0.5)
        # Compare paper by Leonard
        staggered_density_x = []
        staggered_density_y = []
        for j in range(0, dim_y):
            row = []
            for i in range(0, dim_x + 1):
                if (vel_u_data[j][i] > 0):
                    rho_R = rho_C = rho_L = 0.0
                    if(i < dim_x):
                        rho_R = density_data[j][i]
                    if(i > 0):
                        rho_C = density_data[j][i - 1]
                    if(i > 1):
                        rho_L = density_data[j][i - 2]
                    rho = 0.5 * (rho_C + rho_R) - 0.125 * (rho_L + rho_R - 2.0 * rho_C)
                    row.append(rho)
                elif (vel_u_data[j][i] < 0):
                    rho_R = rho_C = rho_FR = 0.0
                    if(i < dim_x):
                        rho_R = density_data[j][i]
                    if(i < dim_x - 1):
                        rho_FR = density_data[j][i + 1]
                    if(i > 0):
                        rho_C = density_data[j][i - 1]
                    rho = 0.5 * (rho_C + rho_R) - 0.125 * (rho_FR + rho_C - 2.0 * rho_R)
                    row.append(rho)
                else:
                    row.append(0.0)
            staggered_density_x.append(row)
        ### Y-Direction is WIP! I set everything to 0 because right now v = 0 anyways!
        for j in range(0, dim_y + 1):
            row = []
            for i in range(0, dim_x):
                if (vel_v_data[j][i] > 0):
                    rho_R = rho_C = rho_L = 0.0
                    if(j < dim_y):
                        rho_R = density_data[j][i]
                    if(j > 0):
                        rho_C = density_data[j - 1][i]
                    if(j > 1):
                        rho_L = density_data[j - 2][i]
                    rho = 0.5 * (rho_C + rho_R) - 0.125 * (rho_L + rho_R - 2.0 * rho_C)
                    row.append(rho)
                elif (vel_v_data[j][i] < 0):
                    rho_R = rho_C = rho_FR = 0.0
                    if(j < dim_y):
                        rho_R = density_data[j][i]
                    if(j < dim_y - 1):
                        rho_FR = density_data[j + 1][i]
                    if(j > 0):
                        rho_C = density_data[j - 1][i]
                    rho = 0.5 * (rho_C + rho_R) - 0.125 * (rho_FR + rho_C - 2.0 * rho_R)
                    row.append(rho)
                else:
                    row.append(0.0)
            staggered_density_y.append(row)


        # Calculate partial derviates dup/dx, dvp/dy and finally dp/dt with velocity (u, v) and density p
        derivates_grid = np.zeros((dim_y, dim_x))
        density_data_step = density_data.copy()

        for j in range(0, dim_y):
            for i in range(0, dim_x):
                # Discretize partial derivates
                rho_x1 = staggered_density_x[j][i]
                rho_x2 = staggered_density_x[j][i + 1]
                #if(i == 0 and j == 1):
                #    print("Density at (0.5, 1) and (1.5, 1): ", rho_x1, ", ", rho_x2)
                u1 = vel_u_data[j][i][0]
                u2 = vel_u_data[j][i + 1][0]
                dupdx = (u2 * rho_x2 - u1 * rho_x1) / delta_x
                #if(i == 0 and j == 1):
                #    print("Density partial derivate at (1, 1): ", dupdx)

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


    def quick_advection_velocity(velocity_field, dt):
        raise NotImplementedError("QUICK for Velocity is WIP!")
    
    import sys
    if 'tf' in sys.argv:
        raise ValueError("This version of QUICK Advection only works with Numpy arrays. Please use the TensorFlow version!")

    if(type == "density"):
        return quick_advection_density(field, velocity_field, dt)
    elif(type == "velocity"):
        return quick_advection_velocity(velocity_field, dt)
