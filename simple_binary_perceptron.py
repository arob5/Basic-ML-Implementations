#
# simple_binary_perceptron.py
# Implementation of a binary classification perceptron
# Modified by: Andrew Roberts
# Last Modified: 7/2/2017
#

#
# TODO : Generalize to be able to handle p features; clean up data generation; change class names from 
# 1, 2 to 0, 1 
#

import random
import numpy as np
import matplotlib.pyplot as plt
import pylab

def main(): 

	LEARNING_RATE = .05

	x1, y1, x2, y2 = generate_data()
	plot_data(x1, y1, x2, y2) 
	num_iterations, weights = learn_weights(x1, y1, x2, y2, LEARNING_RATE) 	
	plot_decision_boundary(weights) 
	print_results(num_iterations, weights) 

def generate_data(): 

	x_class_1 = [.1, .7, .05, .35] 
	y_class_1 = [.1, .1, .5, .9]
	x_class_2 = [-.3, -.9, -.5, .3]
	y_class_2 = [-.3, -.7, -.6, -.4]
	return x_class_1, y_class_1, x_class_2, y_class_2 
'''
	# X data	
	x_class_1 = np.array(random.sample(range(-40, -10), 15)) + np.random.normal(0, 10, 15)   
	x_class_2 = np.array(random.sample(range(10, 40), 15)) + np.random.normal(0, 10, 15)

	# Adjust for any overlap
	for i in range(0, len(x_class_1)): 
		if(x_class_1[i] > 0): 
			x_class_1[i] = 0 
	
	for j in range(0, len(x_class_2)): 
		if(x_class_2[j] <= 0): 
			x_class_2[j] = 1 

	# Y data
	y_class_1 = x_class_1 + np.random.normal(10, 10, 15)
	y_class_2 = x_class_2 + np.random.normal(-10, 10, 15) 
'''


def plot_data(x1, y1, x2, y2): 
	plt.plot(x1, y1, marker="o", linestyle="", color="blue") 
	plt.plot(x2, y2, marker="o", linestyle="", color="red") 
	
	x_concat = np.concatenate((x1,x2))
	y_concat = np.concatenate((y1,y2))
	plt.xlim([x_concat.min()-2, x_concat.max()+2])
	plt.ylim([y_concat.min()-2, y_concat.max()+2])
	

def learn_weights(x1, y1, x2, y2, lr): 
	y_labels_class_1 = np.zeros(len(x1))
	y_labels_class_2 = np.ones(len(x1)) 

	x_train = np.concatenate((x1, x2))
#	y_train_value = np.concatenate((y1, y2))
	y_train_label = np.concatenate((y_labels_class_1, y_labels_class_2))
	
	# Randomly shuffle data
	shuffle_state = np.random.get_state()
	np.random.shuffle(x_train)
#	np.random.set_state(shuffle_state)
#	np.random.shuffle(y_train_value)
	np.random.set_state(shuffle_state)
	np.random.shuffle(y_train_label)

	for i in range(0, len(x_train)): 
		print x_train[i], " ", y_train_label[i]



	# Initialize weights to 0 (dim = 1 "x" + 1 "bias" = 2)  
	weights = np.zeros(2) 

	# Train - Class 2 (ones) if result >= 0 
	found_solution = False
	itr = 0 
	while((not found_solution) and (itr < 20000)):
		itr += 1
		squaredError = 0  
		for i in range(0, len(x_train)):

			# Add bias term (x0=1) to input
			bias = np.array([1])
			x_value = np.array([x_train[i]])
		#	y_value = np.array([y_train_value[i]])
			training_case = np.concatenate((bias, x_value))

			# Calculate predicted output for xi
			pred_output = np.dot(training_case, weights) >= 0  			
			pred_output_test = np.dot(training_case, weights)    			

			# Check if predicted output for xi matches target value
			# If not, adjust weights

			weight_change = lr * (y_train_label[i] - pred_output) * training_case
			weights = weights + weight_change

			squaredError += (y_train_label[i] - pred_output)**2

		if(itr == 20000): 
			print weights 
			print squaredError


		if(squaredError == 0): 
			found_solution = True

	return itr, weights

def plot_decision_boundary(w): 
	print w
	n = pylab.norm(w)
	print n

	ww = w / n
	print ww
	ww1 = [ww[1], -ww[0]]
	print ww1
	ww2 = [-ww[1], ww[0]]
	print ww2
	pylab.plot([ww1[0], ww2[0]], [ww1[1], ww2[1]], "--k")

def print_results(itr, w): 
	print "Converged in: ", itr, " iterations"
	print "Weights: ", w
	plt.show()
	
main()
