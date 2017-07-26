#
# simpleLinearRegression.py
# Python implementation of simple linear regression
# Last modified: 7/24/2017
# Modified By: Andrew Roberts
#

import numpy as np

class LinearRegression():
	""" comment """
	
	def __init__(self):
		coef = None
		intercept = None
		rss = None
	
	@staticmethod
	def calculate_params(X, y, method):
		if method == "basic":
			return LinearRegression.calculate_params_basic(X, y)
		elif method == "vectorized":
			return LinearRegression.calculate_params_vectorized(X, y)
		else:
			raise ValueError("Invalid method argument; must be \"basic\" or \"vectorized\"")
	
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
			params = LinearRegression.calculate_params(X[:,:-1], X[-1], method)
		else:
			params = LinearRegression.calculate_params(X, y, method)

		self.intercept = params[0]
		self.coef = params[1]


fit1 = LinearRegression()
fit1.fit(np.array([0,1,2,3,4,5,6]).reshape(7,1), np.array([0,2,4,6,8,10,12]).reshape(7,1))
print(fit1.coef)
print(fit1.intercept)
'''	
	
fit2 = LinearRegression()
fit2.fit(np.array([0,1,2,3,4,5,6]).reshape(7,1), np.array([0,2,4,6,8,10,12]).reshape(7,1), "vectorized")
print(fit2.coef)
print(fit2.intercept)
			
'''	
	

