import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, initialize_method='xavier'):
        self.initialize_method = initialize_method
        self.weights, self.biases = self.initialize_weights_biases(n_inputs, n_neurons)

    def initialize_weights_biases(self, n_inputs, n_neurons):
        if self.initialize_method == 'xavier':
            # Xavier initialization (Glorot initialization)
            limit = np.sqrt(6 / (n_inputs + n_neurons))
            weights = np.random.uniform(-limit, limit, (n_inputs, n_neurons))
        elif self.initialize_method == 'he':
            # He initialization
            stddev = np.sqrt(2 / n_inputs)
            weights = np.random.normal(0, stddev, (n_inputs, n_neurons))
        elif self.initialize_method == 'gaussian':
            # Gaussian initialization
            weights = np.random.normal(0, 1, (n_inputs, n_neurons))
        else:
            raise ValueError("Unsupported initialization method")

        biases = np.zeros((1, n_neurons))
        return weights, biases

    def forward(self, inputs):
        # Compute linear transformation
        self.output = np.dot(inputs, self.weights) + self.biases
        # Apply ReLU activation function
        self.output = np.maximum(0, self.output)
        return self.output


def plot_decision_boundary(X, y, title):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
    plt.title(title)
    plt.show()


def run_experiment(initialize_method):
    print(f"Running experiment with {initialize_method} initialization")

    # Initialize layers
    layer1 = Layer_Dense(2, 5, initialize_method)
    layer2 = Layer_Dense(5, 3, initialize_method)
    layer3 = Layer_Dense(3, 2, initialize_method)

    # Forward pass
    output1 = layer1.forward(X)
    output2 = layer2.forward(output1)
    output3 = layer3.forward(output2)

    # Plot results
    plot_decision_boundary(X, y, f"{initialize_method.capitalize()} Initialization")
    
    print("Layer 1 Output:")
    print(output1)
    print("\nLayer 2 Output:")
    print(output2)
    print("\nLayer 3 Output:")
    print(output3)


nnfs.init()

# Generate dataset
X, y = spiral_data(samples=100, classes=2)

# Run experiments with different initialization methods
initialize_methods = ['xavier', 'he', 'gaussian']
for method in initialize_methods:
    run_experiment(method)
