import sys

# Add your own project directory to the sys.path
project_dir = "C:/Users/thein/OneDrive/Documents/dd2430"
sys.path.append(project_dir)

import bpy
from path_loader import PathDataLoader

# Use own path to map
paths = PathDataLoader().read_raw(bpy.path.abspath('C:/Users/thein/OneDrive/Documents/eu_city_2x2_macro_306.bin'))

# Create NURBS path
new_curve = bpy.data.curves.new(name="NurbsPath", type='CURVE')
new_curve.dimensions = '3D'  # 3D curve
new_curve.fill_mode = 'FULL'  # Filled curve

# Create a new object to hold the curve data
nurbs_path_obj = bpy.data.objects.new("NurbsPath", new_curve)
bpy.context.collection.objects.link(nurbs_path_obj)

# Access the spline for this curve
spline = new_curve.splines.new(type='NURBS')

# Define the control points for the NURBS path (2 points)
control_points = [(0, 0, 0, 1.0), (1, 1, 1, 1.0)]

# Add the control points to the spline
spline.points.add(len(control_points) - 1)
for i, point in enumerate(spline.points):
    point.co = control_points[i]

# Update the scene to reflect the changes
bpy.context.view_layer.update()


### ORIGINAL SCRIPT ###
# prototype line with only 2 points 
original_line = bpy.data.objects["NurbsPath"]

startpoints = []
endpoints = []

for rx in range(len(paths[0])): # this shows only paths from the first transmission point
    for px in range(len(paths[0][rx])):
        startpoints = [list(point) for point in paths[0][rx][px].points[:-1]]
        endpoints = [list(point) for point in paths[0][rx][px].points[1:]]
        for idx, (s,e) in enumerate(zip(startpoints, endpoints)):
            new_line = original_line.copy()
            new_line.data = original_line.data.copy()
            new_line.name = f"Path-0-{rx}-({px})"

            # set the start- and endpoint 
            new_line.data.splines[0].points[0].co = s + [1.0] # add nurbs weight
            new_line.data.splines[0].points[1].co = e + [1.0]
            bpy.context.collection.objects.link(new_line)        

bpy.context.view_layer.update()

