#
# simpleLinearRegression.py
# Python implementation of simple linear regression for relatively small data sets
# Last modified: 7/28/2017
# Modified By: Andrew Roberts
#

import matplotlib.pyplot as plt
import numpy as np

#
# TODO: Fix error handling; throw exceptions if fit() is not called first
#       Make plotting better (I just threw together something simple)
#       Make multiple plotting figures

class LinearRegression():
	""" comment """
	
	def __init__(self):
		self.X = None
		self.y = None
		self.coef = None
		self.intercept = None
	
	@staticmethod
	def calculate_params(X, y, method):
		""" Calls appropriate function to fit params """
		y = y.reshape(y.shape[0], 1)
	
		if method == "basic":
			return LinearRegression.calculate_params_basic(X, y)
		elif method == "vectorized":
			return LinearRegression.calculate_params_vectorized(X, y)
		else:
			raise ValueError("Invalid method argument; must be \"basic\" or \"vectorized\"")

	def store_data(self, X, y):
		""" Stores data used to fit model """
		self.X = X
		self.y = y

	def predict(self, X):
		""" Returns predicted array of passed in feature array"""
		return np.add(np.multiply(X, self.coef), self.intercept)

	def residuals(self):
		""" Returns array of residuals """ 
		return np.subtract(self.y, self.predict(self.X))

	def rss(self):
		""" Returns the residual sum of squares of the data used to fit the model """
		return (np.square(self.residuals())).sum()
		
	@staticmethod
	def calculate_params_basic(X, y):
		num_obs = float(len(X))
		sum_xy = (X*y).sum()
		sum_x = X.sum()
		sum_xx = (X*X).sum()
		y_bar = (y.sum()) / num_obs
		x_bar = (X.sum()) / num_obs

		b_hat = (sum_xy - (sum_x * y_bar)) / (sum_xx - (sum_x * x_bar))
		a_hat = y_bar - (b_hat * x_bar)
		return np.array([a_hat, b_hat])

	@staticmethod
	def calculate_params_vectorized(X, y):
		X = np.array([np.ones(len(X)), X[:,0]]).T
		y = y.reshape(-1, 1)		
		return np.linalg.solve(X.T.dot(X), X.T.dot(y)).flatten()

	# Add vectorized and gradient descent
	def fit(self, X, y=None, method="basic"):

		if y is None:
			if X.shape[1] == 1:
				raise ValueError("If no y is provided, X must have 2 columns")
			self.store_data(X[:,:1], X[:,1])
		else:
			self.store_data(X, y)
		
		params = LinearRegression.calculate_params(self.X, self.y, method)
		self.intercept = params[0]
		self.coef = params[1]

	def regression_plot(self, regr_line=True):
		""" Plots a scatter plot of data used to fit model
		    Argument: regr_line - boolean, whether or not do plot regression line
			                - Default: True
		"""
		plt.plot(self.X, self.y, marker="o", linestyle="", color="blue")
		plt.plot(self.X, self.predict(self.X), color="red")
		plt.xlabel("X")
		plt.ylabel("y")
		plt.suptitle("Regression Plot")
		plt.show()

	def predictor_plot(self):
		""" Plots the residuals against X """ 
		plt.plot(self.X, self.residuals(), marker="o", linestyle="", color="red")
		plt.plot(self.X, np.zeros(len(self.X)), color="blue")
		plt.xlabel("X")
		plt.ylabel("y - Pred y")
		plt.suptitle("Predictor Plot")
		plt.show()

	def residual_plot(self):
		""" Plots the residuals against the fitted y values """
		fitted_y = self.predict(self.X)
		plt.plot(fitted_y, self.residuals(), marker="o",linestyle="", color="blue")
		plt.plot(fitted_y,  np.zeros(len(fitted_y)), color="red")
		plt.xlabel("Fitted y")
		plt.ylabel("y - Pred y")
		plt.suptitle("Residual Plot")
		plt.show()

fit1 = LinearRegression()

'''
arr = np.array([[0,1,2,3,4,5,6], [0,2,4,6,8,10,12]]).T
arr_X =  np.array([0,1,2,3,4,5,6]).reshape(7,1)
arr_y = np.array([0,2,4,6,8,10,12]).reshape(7,1)

fit1.fit(arr_X, arr_y,  method="basic")
print(fit1.coef)
print(fit1.intercept)
print(fit1.predict(20))
print(fit1.residuals())
print(fit1.rss())
fit1.predictor_plot()
fit1.residual_plot()
'''

def generate_training_data(): 
	X = np.arange(100)
	noise = np.random.uniform(-10,10,size=(100,))
	y = (.5 * X) + (5 + noise)
	
	return X.reshape(100,1), y.reshape(100,1)
	
X,y = generate_training_data()
fit1.fit(X, y)
print "Coef: ", fit1.coef
print "Intercept: ", fit1.intercept
print "RSS: ", fit1.rss()
fit1.regression_plot()
fit1.predictor_plot()
fit1.residual_plot()

