#
# File: xor_forward_prop.py
# Purporse: A basic implementation of a forward pass through a NN using the XOR problem 
# Andrew Roberts
#

import numpy as np 
	
LEARNING_RATE = .05
NUM_LAYERS = 3

def main(): 
	""" Runs helper functions """

	n_neurons_each_layer = [2, 3, 1] 
	X, y = get_input()
	activation = X 

	for i in range(0, NUM_LAYERS-1): 
		print activation
		weights = initialize_weights(n_neurons_each_layer[i], n_neurons_each_layer[i+1])
		print weights
		signal = calculate_signal(activation, weights)
		print signal
		activation = sigmoid(signal) 
		print activation

	print "Output: ", activation

def get_input():
	""" 
	Returns training input for NN:

	X: 4 x 2 feature matrix
	y: 4 x 1 response vector 
	"""
	
	X = np.array([[0, 0], [0, 1], [1, 0], [1,1]], np.int32)
	y = np.array([[0, 1, 1, 0]])
	
	return X, y.T

def calculate_signal(input, weights): 
	""" Return dot product of feature matrix and weights transpose """

	return np.dot(input, weights.T)

def initialize_weights(n_neurons_current, n_neurons_next): 
	""" 
	Returns k x n matrix of random weights in interval [-1.0, 1.0) 
	k = number of neurons in next layer	
	n = number of neurons in current layer

	Hidden layer will have 3 neurons    
	""" 
	np.random.seed(42)
	return 2*np.random.random((n_neurons_next, n_neurons_current)) - 1 

def sigmoid(signal):
	"""
	Passes input to sigmoid function and returns its output
	Will do elementwise operation on matrices
	"""

	exponential = np.exp(signal * -1)
	denominator = np.add(exponential, 1)

	return 1 / denominator 

main()
