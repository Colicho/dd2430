import numpy as np

class PropagationPath:
    def __init__(self):
        self.points = []
        self.interaction_types = []   # 0 - initial transmitter (radio antenna)   1 - final point (user)   2 - specular reflection   3 - diffraction around the edge
        self.path_gain_db = 0   # dB = 10 * log10(I/I0)   -20 dB = 100 times weaker   -90 dB = 1000000000 times weaker
        self.hash = 0   # hashed set of interactions (objects / surfaces)

offset = 0
def read_int():
    global offset, bytes
    offset += 4
    return bytes[offset-4:offset].copy().view(np.int32)[0]
def read_uint():
    global offset, bytes
    offset += 4
    return bytes[offset-4:offset].copy().view(np.uint32)[0]
def read_float():
    global offset, bytes
    offset += 4
    return bytes[offset-4:offset].copy().view(np.float32)[0]

def read_file(filename: str):
    global offset, bytes
    offset = 0
    bytes = np.fromfile(filename, dtype=np.uint8)

    transmitter_count = read_int()
    paths = []
    for tx in range(transmitter_count):
        receiver_count = read_int()
        paths.append([])
        for rx in range(receiver_count):
            path_count = read_int()
            paths[tx].append([])
            for p in range(path_count):
                path = PropagationPath()
                point_count = read_int()
                for _ in range(point_count):
                    x = read_float()
                    y = read_float()
                    z = read_float()
                    path.points.append((x, y, z))
                for _ in range(point_count):
                    interaction = read_int()
                    path.interaction_types.append(interaction)
                path.path_gain_db = read_float()
                path.hash = read_uint()
                paths[tx][rx].append(path)
    return paths

if __name__ == "__main__":
    path_file = 'eu_city_2x2_macro_306.bin'
    paths = read_file(path_file)
