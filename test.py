import numpy as np


a = np.array([-0.0, 1], np.float64)
b = np.array([0, 1], np.float64)
print(a.tobytes() == b.tobytes())
a[0] = 0
print(a)
print(a.tobytes() == b.tobytes())
