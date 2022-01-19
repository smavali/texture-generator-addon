import sys, os, bpy

dir = os.path.dirname(bpy.data.filepath)
if not dir in sys.path:
    sys.path.append(dir)

from main import *
import sampler
from sampler import *

# this next part forces a reload in case you edit the source after you first start the blender session
import importlib

importlib.reload(sampler)

x, y, z = 0, 0, 0
width, height, depth = 10, 10, 1
diameter = .1
spacing = .5

grid_size = 20

for i in range(grid_size):

    diameter1 = i * diameter / 4
    x = i * width
    spacing = .5

    for j in range(grid_size):
        spacing = spacing + .025 * j
        y = j * height

        pd = PoissonDisc(width, height, spacing, 30)
        print(pd)
        center = (x, y)
        pd.sample(center, thresh=.125)
        points_list = pd.samples_t
        print("# of Poisson samples: ", len(pd.samples_t))  # poisson samples

        Generator(base_width=width, base_height=height, base_depth=depth,
                  cone_diameter=diameter1, cone_height=1,
                  cone_shape="", points=pd.samples_t,
                  text=True, join=True, location=(x, y, z)
                  )
