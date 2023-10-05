from path_reader import read_file
paths = read_file('eu_city_2x2_macro_306.bin')

import bpy

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

