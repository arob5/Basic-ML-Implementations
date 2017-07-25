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
		num_obs = None 
		coef = None
		intercept = None
		rss = None
	
	def calculate_params_basic(self, X, y):
		self.num_obs = float(len(X))
		sum_xy = (X*y).sum()
		sum_x = X.sum()
		sum_xx = (X*X).sum()
		y_bar = (y.sum()) / self.num_obs
		x_bar = (X.sum()) / self.num_obs

		b_hat = sum_xy - (sum_x * y_bar)
		a_hat = y_bar - (b_hat * x_bar)
		
		return np.array([a_hat, b_hat])

	# Add vectorized and gradient descent
	def fit(self, X, y=None, method="basic"):
		if y == None:
			if X.shape[1] == 1:
				raise ValueError("If no y is provided, X must have 2 columns")
			params = self.calculate_params_basic(X[:,:-1], X[-1])
		else:
			params = self.calculate_params_basic(X, y)

		self.intercept = params[0]
		self.coef = params[1]


		
fit1 = LinearRegression()
fit1.fit(np.array([0,1,2,3,4,5,6]).reshape(7,1), np.array([0,2,4,6,8,10,12]).reshape(7,1))
print(fit1.num_obs)
print(fit1.coef)
print(fit1.intercept)
	
			
	
	

