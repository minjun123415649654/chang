import  numpy as np
import nnfs
from nnfs.datasets import spiral_data
import  matplotlib.pyplot as plt


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.uniform(0,1,(n_inputs,n_neurons ))
        self.biases = np.random.uniform(0,1,(1,n_neurons))

    def forward(self, inputs):
        return np.dot(inputs, np.array(self.weights)) + self.biases



nnfs.init()

X, y = spiral_data(samples=100, classes=2)
Layer1 = Layer_Dense(2,5)
Layer2 = Layer_Dense(5,3)
Layer3 = Layer_Dense(3,2)
Layer3.forward(Layer2.forward(Layer1.forward(X)))
plt.scatter(X[:, 0], X[:, 1], c = y, cmap = 'brg')
plt.show()

initialize_methods = ['xavier', 'he', 'gaussian']
for method in initialize_methods:
    output1 = layer1.forward(X)
    output2 = layer2.forward(output1)

    print("Layer 1 Output:")
    print(output1)
    print("\nLayer 2 Output:")
    print(output2)