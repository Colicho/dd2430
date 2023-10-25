import numpy as np

class PropagationPath:
    def __init__(self):
        self.points = []
        self.interaction_types = []     # 0 - initial transmitter (radio antenna)   1 - final point (user)   2 - specular reflection   3 - diffraction around the edge
        self.path_gain_db = 0           # dB = 10 * log10(I/I0)   -20 dB = 100 times weaker   -90 dB = 1000000000 times weaker
        self.hash = 0                   # hashed set of interactions (objects / surfaces)

class PathDataLoader:
    def __init__(self):
        self.offset = 0
        self.bytes = 0

    def read_file(self, fileName: str):
        self.offset = 0
        self.bytes = np.fromfile(fileName, dtype=np.uint8)

        transmitter_count = self.__read_int()
        paths = []
        for tx in range(transmitter_count):
            receiver_count = self.__read_int()
            paths.append([])
            for rx in range(receiver_count):
                path_count = self.__read_int()
                paths[tx].append([])
                for p in range(path_count):
                    path = PropagationPath()
                    point_count = self.__read_int()
                    for _ in range(point_count):
                        x = self.__read_float()
                        y = self.__read_float()
                        z = self.__read_float()
                        path.points.append((x, y, z))
                    for _ in range(point_count):
                        interaction = self.__read_int()
                        path.interaction_types.append(interaction)
                    path.path_gain_db = self.__read_float()
                    path.hash = self.__read_uint()
                    paths[tx][rx].append(path)
        return paths
    
    def __read_int(self):
        self.offset += 4
        return self.bytes[self.offset - 4:self.offset].copy().view(np.int32)[0]

    def __read_uint(self):
        self.offset += 4
        return self.bytes[self.offset - 4:self.offset].copy().view(np.uint32)[0]

    def __read_float(self):
        self.offset += 4
        return self.bytes[self.offset - 4:self.offset].copy().view(np.float32)[0]