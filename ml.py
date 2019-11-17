import numpy as np
import random

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
  error = np.subtract(wanted_output, output_result))
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



  

network = Neural_network([2, 2, 1])




training_set = [[np.array([[1], [1]]), [[0]]], [np.array([[0], [0]]), [[0]]], [np.array([[0], [1]]), [[1]]], [np.array([[1], [0]]), [[1]]]]
choice = random.choice(training_set)
for i in range(50000):
  choice = random.choice(training_set)
  train(choice[0], network, choice[1])


print(output(np.array([[1], [1]]), network))
print(output(np.array([[0], [0]]), network))
print(output(np.array([[0], [1]]), network))
print(output(np.array([[1], [0]]), network))



	








