import sys
if not 'tf' in sys.argv:
    raise RuntimeError("This simulation can only be run in TensorFlow-Mode!")
from phi.tf.flow import *  # Use TensorFlow
MODE = 'TensorFlow'
RESOLUTION = [int(sys.argv[1])] * 2 if len(sys.argv) > 1 and __name__ == '__main__' else [128] * 2
DESCRIPTION = "Basic fluid test that runs QUICK Scheme with CUDA"


class CUDAFlow(App):
    def __init__(self):
        App.__init__(self, 'CUDA Flow', DESCRIPTION, summary='fluid' + 'x'.join([str(d) for d in RESOLUTION]), framerate=20) 




print("CUDA TF QUICK Advection")
