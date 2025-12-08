import numpy as np
class fnnlayer:
    def __init__(self, input_size, output_size, activation_function):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_function = activation_function
        self.weights = self.initialize_weights(input_size, output_size)
        self.biases = self.initialize_biases(output_size)

    def initialize_weights(self, input_size, output_size):
        # Initialize weights with small random values
        return np.random.randn(input_size, output_size) * 0.01

    def initialize_biases(self, output_size):
        # Initialize biases to zero
        return np.zeros((1, output_size))

    def forward(self, inputs):
        # Compute the linear combination of inputs and weights
        z = np.dot(inputs, self.weights) + self.biases
        # Apply the activation function
        return self.activation_function(z)