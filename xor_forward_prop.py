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
	weights = initialize_weights(n_neurons_each_layer[0], n_neurons_each_layer[1])
	signal = calculate_signal(input, weights)
	activation = activate_signal(input, weights) 

def get_input():
	""" 
	Returns training input for NN:

	X: 4 x 2 feature matrix
	y: 4 x 1 response vector 
	"""
	
	X = np.array([[0, 0], [0, 1], [1, 0], [1,1]], np.int32)
	y = np.array([[0, 1, 1, 0]])
	
	return X, y.T

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

	exponential = -1 * np.exp(signal)
	denominator = np.add(exponential, 1)

	return 1 / denominator 






	


main()
