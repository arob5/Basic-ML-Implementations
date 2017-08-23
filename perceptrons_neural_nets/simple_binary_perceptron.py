#
# simple_binary_perceptron.py
# Implementation of a binary classification perceptron
# Modified by: Andrew Roberts
# Last Modified: 8/23/2017
#

import random
import numpy as np
import matplotlib.pyplot as plt
import pylab

def main(): 
	LEARNING_RATE = .05
	X_train = generate_data()
	train(X_train, LEARNING_RATE) 	

def generate_data(): 
	return np.random.randint(low=-20, high=20, size=(100,2))

def train(X, lr): 
	bias_ones = np.ones(shape=(len(X), 1))
	X = np.concatenate((bias_ones, X), axis=1)
	np.random.shuffle(X)
	W = np.random.normal(size=len(X[0])) 
	
	for i in range(100):
		classified_1 = []
		classified_0 = [] 

		error_sq = 0
		for example in X:
			if example[2] >= example[1]:
				target = 1
			else:
				target = 0

			y_pred = calc_output(example, W)

			if y_pred == 1:
				classified_1.append(list(example))
			elif y_pred == 0:
				classified_0.append(list(example))

			error = target - y_pred
			error_sq += (error*error)
			
			W = adjust_weights(error, example, W, lr)

		print("Epoch ", i+1, "----- Squared Error =", error_sq)
		if error_sq == 0:
			break

	plot_results(classified_1, classified_0)

	
def calc_output(x, w):
	prod = x @ w	

	if prod >= 0:
		return 1

	return 0

def adjust_weights(error, x, w, lr):
	w_delta = error * x * lr
	return w + w_delta 

def plot_results(x1, x0):
	x1 = np.array(x1)
	x0 = np.array(x0)

	plt.plot(x1[:,1], x1[:,2], marker="o", linestyle="", color="blue") 	
	plt.plot(x0[:,1], x0[:,2], marker="o", linestyle="", color="red") 	

	plt.show()

main()
