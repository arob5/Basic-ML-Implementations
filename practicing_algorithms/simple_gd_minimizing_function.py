#
# simple_gd_minimizing_function.py
# Program that minimizes f(x) = x^2 - 2 using a simple batch gradient descent
# Modified By: Andrew Roberts
# Last Modified: 7/23/2017
#

import time
import matplotlib.pyplot as plt
import numpy as np
from random import randint 

def main():
	# Define learning rate 
	learning_rate = .1

	# Randomly initialize x
	x = float(randint(-40, 40)) 
	print "Starting value for x: ", x

	# Run Gradient Descent and print results
	output, x_plot, y_plot = gd(learning_rate, x)
	print_results(output)
	
	# Plot Function
	plot_function(x_plot, y_plot)
	
def gd(learning_rate, x): 
	# Define variable to count iterations
	iteration = 0 

	# To store data for graph
	x_for_graph = []
	y_for_graph = []

	# Loop until convergence (5000 iterations or gradient smaller than .0001)
	t0 = time.time()
	for i in range(0, 1000): 
		
		# Store x-y data
		x_for_graph.append(x)
		y_for_graph.append((x*x)/2)

		# Count iterations
		iteration += 1 
		
		# Calculate gradient
		gradient = float(2 * x)

		# Break if absolute value of gradient < .0001
		if(abs(gradient) < .0001):
			break

		# Update x
		x = x - learning_rate * gradient
	t1 = time.time()

	results = []
	results.append(x)
	results.append(iteration)
	results.append(t1-t0)
	return results, x_for_graph, y_for_graph 

def print_results(output): 
	print "Converged to {0[0]} in {0[1]} iteration(s)".format(output)
	print "Time to complete: {0[2]}".format(output)

def plot_function(x_plot, y_plot): 
	x_values = np.linspace(-40, 40, 80)
	y_values = (x_values)**2 / 2 
	plt.plot(x_values, y_values, color="blue")
	plt.plot(x_plot, y_plot, marker="o", linestyle="", color="red")
	plt.show()

# Call to main()
main()
