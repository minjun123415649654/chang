import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# Dense Layer Class
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


# Activation ReLu Class
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


# Activation Softmax Class
class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


# Loss Categorical Crossentropy Class with forward method
class Loss_CategoricalCrossentropy:
    def forward(self, predictions, targets):
        samples = len(predictions)
        predictions_clipped = np.clip(predictions, 1e-7, 1 - 1e-7)
        if len(targets.shape) == 1:
            correct_confidences = predictions_clipped[range(samples), targets]
        elif len(targets.shape) == 2:
            correct_confidences = np.sum(predictions_clipped * targets, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return np.mean(negative_log_likelihoods)


# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)

# Forward pass through first Dense layer
dense1.forward(X)

# Activation function - ReLU
activation1 = Activation_ReLU()
activation1.forward(dense1.output)

# Create second Dense layer with 3 output features
dense2 = Layer_Dense(3, 3)

# Forward pass through second Dense layer
dense2.forward(activation1.output)

# Activation function - Softmax
activation2 = Activation_Softmax()
activation2.forward(dense2.output)

# Loss function
loss_function = Loss_CategoricalCrossentropy()

# Calculate loss (using Softmax activation outputs)
loss = loss_function.forward(activation2.output, y)

# Print loss
print('Loss:', loss)
