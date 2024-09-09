
import numpy as np
inputs = [[2.0, 3.0, 4.0, 2.5],
          [3.0, 4.0, -2.0, 1.0],
          [-1.1,2.5, 3.4, -0.5],
          ]

weights = [[0.4, 0.7, -0.1, 1.0],
           [0.7, -0.21, 0.25, -0.1],
           [-0.76, -0.37, 0.57, 0.86]]

biases = [2.0, 3.0, 0.5]

layers_outputs = np.dot(inputs, np.array(weights).T) +biases

print(layers_outputs)

weights_1 = [[0.2,1.0,3.0]]

biases_1 = [1.0]

layers_outputs_1 = np.dot(layers_outputs, np.array(weights_1).T) +biases_1
print(layers_outputs_1)