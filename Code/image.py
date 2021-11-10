import numpy as np
import mlmath


class Image:
    def __init__(self, data, label: int, width: int, height: int):
        self.label = label
        self.data = data
        self.width = width
        self.height = height
        self.normalizedData = []

        for d in self.data:
            self.normalizedData.append(mlmath.normalize(d, 0, 255))

    def expectedVector(self):
        v = np.zeros(10)
        v[self.label] = 1
        return v
