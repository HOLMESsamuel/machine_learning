import numpy as np

def sigmoid(x):
	return (1/(1+np.exp(-x)))

class Layer:
  def __init__(self, size, previous_size):
    self.size = size
    self.previous_size = previous_size
    self.weights = np.ones((size, previous_size))
    self.biases = np.ones((size,1))
    self.values = np.zeros((size,1))

class Neural_network:
  def __init__(self, dimensions):
    layers = []
    input_layer = Layer(dimensions[0], 0)
    layers.append(input_layer)
    for i in range(1, len(dimensions)):
      new_layer = Layer(dimensions[i], dimensions[i-1])
      layers.append(new_layer)
    self.layers = layers

  def enter_input(self, input_values):
    for i in range(len(input_values)):
      self.layers[0].values[i][0] = input_values[i]




def next_layer(previous_layer, current_layer):
  current_layer.values = np.add(np.dot(current_layer.weights, previous_layer.values), current_layer.biases)
  current_layer.values = sigmoid(current_layer.values)


def output(input, network):
  steps = len(network.layers)
  network.enter_input(input)
  for i in range(1, steps):
    next_layer(network.layers[i-1], network.layers[i])
  return network.layers[steps-1].values

  

network = Neural_network([2, 2, 1])

print(output([1, 1], network))
	








