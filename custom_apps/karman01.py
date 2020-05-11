import sys
if 'tf' in sys.argv:
    from phi.tf.flow import *  # Use TensorFlow
    MODE = 'TensorFlow'
else:
    from phi.flow import *  # Use NumPy
    MODE = 'NumPy'


class Karman(App):

    def __init__(self, size):
        App.__init__(self, 'Unsteady wake flow', 
                    'Karman vortex street behind a cylinder for varying Reynolds numbers',
                     summary='karman' + 'x'.join([str(d) for d in size]), stride=20)
        #self.physics = IncompressibleFlow()
        self.physics = SimpleFlowPhysics()
        smoke = self.smoke = world.add( Fluid(Domain(size, box=box[0:200, 0:100], boundaries=OPEN), buoyancy_factor=0.), physics=self.physics)

        world.add( Inflow(box[5:10, 25:75 ]) ) # yx , sm
        world.add( Obstacle(Sphere([50,50], 10)) )

        self.add_field('Density', lambda: smoke.density)
        self.add_field('Velocity', lambda: smoke.velocity)
        self.add_field('Domain', lambda: obstacle_mask(smoke).at(smoke.density))

        self.Re = EditableFloat('Re', 1e6, (1e4, 1e8)) # Reynolds nr
        self.action_reset()

    def action_reset(self):
        self.steps = 0
        self.smoke.density = self.smoke.velocity = 0
        vn = self.smoke.velocity.staggered_tensor()

        # warm start - initialize flow to 1 along y everywhere
        vn[..., 0] = 1.0
        # modify x, poke sideways to trigger instability
        vn[...,vn.shape[1]//2+10:vn.shape[1]//2+20, vn.shape[2]//2-2:vn.shape[2]//2+2, 1] = 1.0 

        # set physical size via self.smoke.domain.box or self.smoke.velocity.box, as done here
        # (important for sanity check when updating the smoke state)
        self.smoke.velocity = StaggeredGrid(unstack_staggered_tensor(vn), self.smoke.velocity.box ) 
        #self.smoke.velocity = StaggeredGrid(vn)

    def step(self):
        # in contrast to action_reset, use the staggered_tensor here - just for demonstration purposes
        if 1:
            vn = self.smoke.velocity.data[0].data
            vn[..., 0:2, 0:vn.shape[2]-1, 0] = 1.0
            vn[..., 0:vn.shape[1], 0:1,   0] = 1.0
            vn[..., 0:vn.shape[1], -1:,   0] = 1.0  
            self.smoke.velocity = StaggeredGrid([vn, self.smoke.velocity.data[1].data], self.smoke.velocity.box ) 
        else:
            vn = self.smoke.velocity.staggered_tensor()
            vn[..., 0:2, 0:vn.shape[2]-1, 0] = 1.0
            vn[..., 0:vn.shape[1], 0:1,   0] = 1.0
            vn[..., 0:vn.shape[1], -2:,   0] = 1.0 # warning - larger by 1, use "-2:" range
            # set physical size via self.smoke.domain.box or self.smoke.velocity.box, as done here
            # (important for sanity check when updating the smoke state)
            self.smoke.velocity = StaggeredGrid(unstack_staggered_tensor(vn), self.smoke.velocity.box ) 

        if 1: # viscosity
            # assume 1 unit for cylinder, compute diffusion strength from Re
            alpha = 1./max(self.Re,1e4) * vn.shape[2] * vn.shape[2] # *dt == 1 , Re !

            vel = self.smoke.velocity.data
            cy=diffuse( CenteredGrid(vel[0].data), alpha ) 
            cx=diffuse( CenteredGrid(vel[1].data), alpha ) 
            self.smoke.velocity = StaggeredGrid([cy.data,cx.data], self.smoke.velocity.box)

        world.step()
        if 0 and self.steps%10==9:
            save_img(self.smoke.density.data, 10000., "./tk5c_%04d.png" % (self.steps//10) )


size = int(sys.argv[1]) if len(sys.argv) > 1 else 48
show( Karman([size*2, size]), 
     display=('Density', 'Velocity'), framerate=2)
