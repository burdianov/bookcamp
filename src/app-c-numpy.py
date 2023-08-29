import numpy as np

zeros = np.zeros(10)
ones = np.ones(10)
twos = ones * 2

array = np.full(10, 3)
array = np.repeat(3, 10)

array = np.repeat([0.0, 1.0], 5)
array = np.repeat([0.0, 1.0], [2, 3])

elements = [1, 2, 3, 4]
array = np.array(elements)

thresholds = np.linspace(0, 1, 11)

zeros = np.zeros((5, 2), dtype=np.float32)
