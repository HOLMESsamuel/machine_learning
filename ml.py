import numpy as np

class Layer:
  def __init__(self, size, previous_size):
    self.size = size
    self.previous_size = previous_size
    self.weights = np.ones((size, previous_size))
    self.biases = np.ones((size,1))
    self.values = np.zeros((size,1))

def sigmoid(x):
	return (1/(1+np.exp(-x)))


def next_layer(previous_layer, current_layer):
	current_layer.values = np.add(np.dot(current_layer.weights, previous_layer.values), current_layer.biases)


def output(layers):
	steps = len(layers)
	for i in range(1, steps):
		next_layer(layers[i-1], layers[i])
	return layers[steps-1].values

def create_layers(input_values, format):
  layers = []
  input_layer = Layer(len(input_values), 0)
  for i in range(len(input_values)):
    input_layer.values[i][0] = input_values[i]
  layers.append(input_layer)
  for i in range(1, len(format)):
    new_layer = Layer(format[i], format[i-1])
    layers.append(new_layer)
  return layers


print(output(create_layers([1, 1], [2, 2, 2])))
	








