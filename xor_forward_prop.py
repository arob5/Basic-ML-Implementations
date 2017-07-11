#
# File: xor_forward_prop.py
# Purporse: A basic implementation of a forward pass through a NN using the XOR problem 
# Andrew Roberts
#

import numpy as np 
	
LEARNING_RATE = .05

def main(): 
	""" Runs helper functions """

	X, y = get_input()
	weights = initialize_weights()
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

def initialize_weights(): 
	""" Returns random weights in interval [-1.0, 1.0) """ 

	np.random.seed(42)
	return 2*np.random.random((3,2)) - 1 

def sigmoid(signal):
	"""
	Passes input to sigmoid function and returns its output
	Will do elementwise operation on matrices
	"""

	exponential = -1 * np.exp(signal)
	denominator = np.add(exponential, 1)

	return 1 / denominator 






	


main()
