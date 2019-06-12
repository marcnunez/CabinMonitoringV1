import numpy as np

class Keypoint:

    def __init__(self, x: float, y: float, v: float):
        self.x = x
        self.y = y
        self.v = v

    def get_coordinates(self) -> np.array:
        return np.array(self.x, self.y)
