
import numpy as np
inputs = [[2.0, 3.0, 4.0, 2.5],
          [3.0, 4.0, -2.0, 1.0],
          [-1.1,2.5, 3.4, -0.5],
          [1.2, 3.4, 4.5 , 6.7],
          [2.3, 3.5, 4.7, 6.8]]

weights = [[0.4, 0.7, -0.1, 1.0],
           [0.7, -0.21, 0.25, -0.1],
           [-0.76, -0.37, 0.57, 0.86]]

biases = [2.0, 3.0, 0.5]

layers_outputs = np.dot(inputs, np.array(weights).T) +biases

print(layers_outputs)
