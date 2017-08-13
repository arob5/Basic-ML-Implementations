#
# simpleLinearRegression.py
# Python implementation of simple linear regression for relatively small data sets
# Last modified: 8/13/2017
# Modified By: Andrew Roberts
#

import matplotlib.pyplot as plt
import numpy as np

#
# TODO: Fix error handling; throw exceptions if fit() is not called first
#       Make plotting better (I just threw together something simple)
#       Make multiple plotting figures

class LinearRegression():
	""" Provides functionality to fit and predict using a simple linear regression model"""
	
	def __init__(self):
		self.X = None
		self.y = None
		self.coef = None
		self.intercept = None
	
	@staticmethod
	def calculate_params(X, y, method):
		""" Calls appropriate function to fit params 

		Args:
		    X (Numpy Array)- Feature matrix
		    y (Numpy Array) - Response vector

		Raises ValueError for invalid input
		"""
		y = y.reshape(y.shape[0], 1)
	
		if method == "basic":
			return LinearRegression.calculate_params_basic(X, y)
		elif method == "vectorized":
			return LinearRegression.calculate_params_vectorized(X, y)
		else:
			raise ValueError("Invalid method argument; must be \"basic\" or \"vectorized\"")

	def store_data(self, X, y):
		""" Stores data used to fit model 

		Args:
		    X (Numpy array) - Feature matrix
		    y (Numpy array) - Response vector
		"""
		self.X = X
		self.y = y

	def predict(self, X):
		""" Predicts response given a feature input
		
		Args:
		    X (Numpy Array) - Feature matrix to be used for prediction

		Returns:
		    Numpy Array - Predicted response values
		"""
		return np.add(np.multiply(X, self.coef), self.intercept)

	def residuals(self):
		""" Returns array of residuals """ 
		return np.subtract(self.y, self.predict(self.X))

	def rss(self):
		""" Returns the residual sum of squares of the data used to fit the model """
		return (np.square(self.residuals())).sum()
		
	@staticmethod
	def calculate_params_basic(X, y):
		""" One method to calculate the least squares estimators

		Args:
		    X (Numpy Array) - Feature matrix (training data)
		    y (Numpy Array) - Response vector (training data)

		Returns:
		    list : The least squares estimators
		"""
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
		""" A vectorized implementation to find the least squares estimators
	
		Args:
		    X (Numpy Array) - Feature matrix (training data)
		    y (Numpy Array) - Response vector (training data)

		Returns:
		    Array : The least squares estimators    
		"""
		X = np.array([np.ones(len(X)), X[:,0]]).T
		y = y.reshape(-1, 1)		
		return np.linalg.solve(X.T.dot(X), X.T.dot(y)).flatten()

	def fit(self, X, y=None, method="basic"):
		""" Parses training data and calls functions to fit model
		    If no y is passed, then it is assumed that y is the last
		    column of X

		Args:
		    X (Numpy array) - Feature matrix (training data)
		    y (Numpy array) - Response vector (training data); Default: None
		
		Raises ValueError for invalid input
		"""

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

		Args:
		    regr_line (bool): Include regression line in plot; Default: True
		"""
		plt.plot(self.X, self.y, marker="o", linestyle="", color="blue")
		plt.plot(self.X, self.predict(self.X), color="red")
		plt.xlabel("X")
		plt.ylabel("y")
		plt.suptitle("Regression Plot")
		plt.show()

	def predictor_plot(self):
		""" Plots the residuals against X""" 
		plt.plot(self.X, self.residuals(), marker="o", linestyle="", color="red")
		plt.plot(self.X, np.zeros(len(self.X)), color="blue")
		plt.xlabel("X")
		plt.ylabel("y - Pred y")
		plt.suptitle("Predictor Plot")
		plt.show()

	def residual_plot(self):
		""" Plots the residuals against the fitted y values"""
		fitted_y = self.predict(self.X)
		plt.plot(fitted_y, self.residuals(), marker="o",linestyle="", color="blue")
		plt.plot(fitted_y,  np.zeros(len(fitted_y)), color="red")
		plt.xlabel("Fitted y")
		plt.ylabel("y - Pred y")
		plt.suptitle("Residual Plot")
		plt.show()

