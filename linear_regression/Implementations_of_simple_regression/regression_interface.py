#
# regression_interface.py
# A simple interface to interact with the simple linear regression class
# Last Modified: 8/13/2017
# Modified By: Andrew Roberts
#

import simple_linear_regression as slr
import numpy as np

def main():
	X,y = generate_training_data()
	fit1 = slr.LinearRegression()
	fit1.fit(X, y)
	
	usr_input = ""
	while usr_input != "q":
		usr_input = input(">> ")
		valid_input = ensure_valid_input(usr_input)
		
		if usr_input == "q":
			break

		if valid_input:
			parse_input(fit1, usr_input)
		else:
			print("Unrecognized command")

def generate_training_data(): 
	""" Generates random linear data with some noise
	
	Returns:
	    X values, y values (both Numpy arrays)
	"""
	X = np.arange(100)
	noise = np.random.uniform(-10,10,size=(100,))
	slope = np.random.uniform(low=-10, high=10)
	y_int = np.random.uniform(low=-20, high=20)
	y = (slope * X) + (y_int + noise)
	
	return X.reshape(100,1), y.reshape(100,1)

def ensure_valid_input(input):
	""" Returns whether input is valid or not

	Args:
	    input (str): User input

	Returns:
	    bool: Input is valid
	"""
	commands = ["q", "coef", "inter", "rss", "reg-plot", "pred-plot", "resid-plot"]
	return input in commands

def parse_input(fit1, input):
	""" Performs appropriate action based on user input

	Args:
	   fit1 (LinearRegression object): Fit model
	   input (str): User input
	"""
	if input == "coef":
		print("Coef: ", fit1.coef)
	elif input == "inter":
		print("Intercept: ", fit1.intercept)
	elif input == "rss":
		print("RSS: ", fit1.rss())
	elif input == "reg-plot":
		fit1.regression_plot()
	elif input == "pred-plot":
		fit1.predictor_plot()
	elif input == "resid-plot":
		fit1.residual_plot()
	
main()
