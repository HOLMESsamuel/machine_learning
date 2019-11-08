import numpy as np

class Layer:
  def __init__(self, size, previous_size):
    self.size = size
    self.previous_size = previous_size
    self.weights = np.zeros((size, previous_size))
    self.biases = np.zeros((size,1))
    self.values = np.zeros((size,1))

def sigmoid(x):
	return (1/(1+np.exp(-x)))


def next_layer(previous_layer, current_layer):
	current_layer.values = np.add(np.dot(current_layer.weights, previous_layer.values), current_layer.biases)


def output(layers):
	steps = len(layers)
	for i in range(1, steps):
		next_layer(layers[i-1], layers[i])
	return output_layer.values
	

input_layer = Layer(2, 0)
input_layer.values = np.array([[1], [1]])

layer1 = Layer(2, 2)

output_layer = Layer(2, 2)

layers = [input_layer, layer1, output_layer]

print(output(layers))






