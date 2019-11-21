import numpy as np
import random
import matplotlib.pyplot as plt

def sigmoid(x):
	return (1/(1+np.exp(-x)))

def d_sigmoid(x):
  return x*(1-x)

class Layer:
  def __init__(self, size, previous_size):
    self.size = size
    self.previous_size = previous_size
    self.weights = np.random.rand(size, previous_size)
    self.biases = np.random.rand(size,1)
    self.values = np.zeros((size,1))

class Neural_network:
  def __init__(self, dimensions):
    self.dimensions = dimensions
    self.learning_rate = 0.1
    layers = []
    input_layer = Layer(dimensions[0], 0)
    layers.append(input_layer)
    for i in range(1, len(dimensions)):
      new_layer = Layer(dimensions[i], dimensions[i-1])
      layers.append(new_layer)
    self.layers = layers
  
  def display(self):
    for i in range(len(self.layers)):
      print(self.layers[i].weights, self.layers[i].values, self.layers[i].biases)
  
  def draw(self, left, right, bottom, top):
    '''
    :parameters:
        - left : float
            The center of the leftmost node(s) will be placed here
        - right : float
            The center of the rightmost node(s) will be placed here
        - bottom : float
            The center of the bottommost node(s) will be placed here
        - top : float
            The center of the topmost node(s) will be placed here
    '''
    fig = plt.figure(figsize=(12, 12))
    ax = fig.gca()
    ax.axis('off')
    n_layers = len(self.dimensions)
    v_spacing = (top - bottom)/float(max(self.dimensions))
    h_spacing = (right - left)/float(len(self.dimensions) - 1)
    # Nodes
    for n, layer_size in enumerate(self.dimensions):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in xrange(layer_size):
            circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,
                                color='w', ec='k', zorder=4)
            ax.add_artist(circle)
    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(self.dimensions[:-1], self.dimensions[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in xrange(layer_size_a):
            for o in xrange(layer_size_b):
                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k')
                ax.add_artist(line)
    fig.savefig('nn.png')




def enter_input(network, input_values):
    network.layers[0].values = input_values


def next_layer(previous_layer, current_layer):
  current_layer.values = np.add(np.dot(current_layer.weights, previous_layer.values), current_layer.biases)
  current_layer.values = sigmoid(current_layer.values)


def feedforward(input, network):
  steps = len(network.layers)
  enter_input(network, input)
  for i in range(1, steps):
    next_layer(network.layers[i-1], network.layers[i])
  return network.layers


def output(input, network):
  steps = len(network.layers)
  enter_input(network, input)
  for i in range(1, steps):
    next_layer(network.layers[i-1], network.layers[i])
  return network.layers[-1].values

def error(input, network, wanted_output):
  errors = []
  output_result = output(input, network)
  error = np.subtract(wanted_output, output_result)
  errors.append(error)
  for i in range(1, len(network.layers)-1):
    error = np.dot(network.layers[len(network.layers)-i].weights.transpose(), error)
    errors.append(error)
  return errors

def delta(input, network, wanted_output):
  errors = error(input, network, wanted_output)
  fed_layers = feedforward(input, network)
  fed_layers.reverse()
  weight_deltas = []
  biases_deltas = []
  for i in range(len(network.layers)-1):
    gradients = network.learning_rate * np.multiply(errors[i], d_sigmoid(fed_layers[i].values))
    biases_deltas.append(gradients)
    deltas = np.dot(gradients, fed_layers[i+1].values.transpose())
    weight_deltas.append(deltas)
  fed_layers.reverse()
  return [weight_deltas, biases_deltas]

def train(input, network, wanted_output):
  deltas = delta(input, network, wanted_output)
  weight_deltas, biases_deltas = deltas[0], deltas[1]
  weight_deltas.reverse()
  biases_deltas.reverse()
  for i in range(len(network.layers)-1):
    network.layers[i+1].weights = np.add(network.layers[i+1].weights, weight_deltas[i])
    network.layers[i+1].biases = np.add(network.layers[i+1].biases, biases_deltas[i])



  

network = Neural_network([2, 4, 1])




training_set = [[np.array([[1], [1]]), [[0]]], [np.array([[0], [0]]), [[0]]], [np.array([[0], [1]]), [[1]]], [np.array([[1], [0]]), [[1]]]]
choice = random.choice(training_set)
for i in range(50000):
  choice = random.choice(training_set)
  train(choice[0], network, choice[1])

network.draw(.1, .9, .1, .9)
print(output(np.array([[1], [1]]), network))
print(output(np.array([[0], [0]]), network))
print(output(np.array([[0], [1]]), network))
print(output(np.array([[1], [0]]), network))



	








