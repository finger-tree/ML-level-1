#just an input neuron, an intermediate neuron, and an output neuron
class NeuralChain:
    def __init__(self, input_neuron, intermediate_neuron, output_neuron):
        self.input_neuron = input_neuron
        self.intermediate_neuron = intermediate_neuron
        self.output_neuron = output_neuron

    def forward(self, inputs):
        # Pass inputs through the input neuron
        intermediate_output = self.input_neuron.forward(inputs)
        # Pass the result through the intermediate neuron
        output = self.intermediate_neuron.forward(intermediate_output)
        # Finally, pass through the output neuron
        final_output = self.output_neuron.forward(output)
        return final_output
    
    def backward(self, loss_gradient, learning_rate):
        # Backpropagate through the output neuron
        output_gradient = self.output_neuron.backward(loss_gradient, learning_rate)
        # Backpropagate through the intermediate neuron
        intermediate_gradient = self.intermediate_neuron.backward(output_gradient, learning_rate)
        # Backpropagate through the input neuron
        self.input_neuron.backward(intermediate_gradient, learning_rate)

# Example usage:
if __name__ == "__main__":
    import numpy as np
    # Dummy neuron class for demonstration
    class DummyNeuron:
        def forward(self, inputs):
            return inputs * 2  # Simple operation for demonstration

        def backward(self, gradient):
            return gradient * 0.5  # Simple operation for demonstration
    
    input_neuron = DummyNeuron()
    intermediate_neuron = DummyNeuron()
    output_neuron = DummyNeuron()
    neural_chain = NeuralChain(input_neuron, intermediate_neuron, output_neuron)    
    inputs = np.array([1.0, 2.0, 3.0])
    print("Inputs:", inputs)
    output = neural_chain.forward(inputs)
    print("Final output:", output)
    loss_gradient = np.array([0.1, 0.2, 0.3])
    neural_chain.backward(loss_gradient)
    print("Backpropagation completed.")