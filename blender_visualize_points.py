import sys

# Add your own project directory to the sys.path
project_dir = "C:/Users/thein/OneDrive/Documents/dd2430"
sys.path.append(project_dir)

import bpy
from path_loader import PathDataLoader

# Use own path to map
paths = PathDataLoader().read_raw(bpy.path.abspath('C:/Users/thein/OneDrive/Documents/eu_city_2x2_macro_306.bin'))

startpoints = []
endpoints = []
red_material = bpy.data.materials.new(name="PointColor")
red_material.diffuse_color = (1.0, 0.0, 0.0, 1.0)

blue_material = bpy.data.materials.new(name="PointColor")
blue_material.diffuse_color = (0.0, 0.0, 1.0, 1.0)
transmitter_set = set()
receiver_set = set()

for tx in range(len(paths)):
    for rx in range(len(paths[tx])):
        if len(paths[tx][rx]) != 0:
            points = paths[tx][rx][0].points
            startpoint = points[0]
            endpoint = points[-1]
            transmitter_set.add(startpoint)
            break


for tx in range(len(paths))[:1]:
    for rx in range(len(paths[tx])):  # this shows only paths from the first transmission point
        if len(paths[tx][rx]) != 0:
            points = paths[tx][rx][0].points
            startpoint = points[0]
            endpoint = points[-1]
            transmitter_set.add(startpoint)
            receiver_set.add(endpoint)


for i, transmitter in enumerate(transmitter_set):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=transmitter)
    sphere_object = bpy.context.active_object
    sphere_object.name = f"Transmitter_{i}"
    bpy.context.object.data.materials.append(red_material)
    # bpy.context.collection.objects.link(sphere_object)

for i, receiver in enumerate(receiver_set):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=receiver)
    sphere_object = bpy.context.active_object
    sphere_object.name = f"Receiver_{i}"
    bpy.context.object.data.materials.append(blue_material)
    # bpy.context.collection.objects.link(sphere_object)

bpy.context.view_layer.update()