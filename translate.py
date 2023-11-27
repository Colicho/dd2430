import numpy as np

def translate_paths(patches, rotate=True):
    """
    This function translates the paths to the origin, rotates them so that the last point is on the x-axis.
    Optionally do not rotate the paths and only translate.
    """
    num_paths_in_patches = []
    for i in patches:
        num_paths_in_patches.append(len(i))


    paths = [path for patch in patches for path in patch]


    for i in range(len(paths)):
        path_array = np.array(paths[i])
        # Extracting the points of the path
        path_points = path_array[:15].reshape((5, 3))

        #Translate all points to start at (0,0,0)
        last_point_index = np.where(np.any(path_points != 0, axis=1))[0][-1]
        path_points[:last_point_index + 1] -= path_points[0]

        if rotate:

            current_direction = path_points[last_point_index] / np.linalg.norm(path_points[last_point_index])

            # Compute the rotation axis
            rotation_axis = np.cross(current_direction, [1, 0, 0])
            rotation_axis /= np.linalg.norm(rotation_axis)

            # Compute the rotation angle
            cos_angle = np.dot(current_direction, [1, 0, 0])
            angle = np.arccos(cos_angle)

            # Create a rotation matrix or quaternion
            rotation_matrix = rotation_matrix_from_axis_angle(rotation_axis, angle)

            # Apply the rotation to all points in the path
            path_points = np.dot(rotation_matrix, np.transpose(path_points)).T


        # Update the matrix with the rotated path
        path_array[:15] = path_points.flatten()
        paths[i] = path_array

    # Re-create the 3D list
    patches = [[] for _ in range(len(num_paths_in_patches))]
    c = 0
    for i in range(len(num_paths_in_patches)):
        for j in range(num_paths_in_patches[i]):
            patches[i].append(paths[c])
            c += 1

    return patches



def rotation_matrix_from_axis_angle(axis, angle):
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c
    x, y, z = axis

    rotation_matrix = np.array([
        [t*x*x + c, t*x*y - z*s, t*x*z + y*s],
        [t*x*y + z*s, t*y*y + c, t*y*z - x*s],
        [t*x*z - y*s, t*y*z + x*s, t*z*z + c]
    ])

    return rotation_matrix