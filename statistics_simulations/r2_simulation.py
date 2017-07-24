#
# r2_simulation.py
# Purpose: A simulation showing that R^2 will increase as features are added, even if the features are 
#          random noise
# Andrew Roberts
#

from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

def main(): 
	X, y = generate_training_data()

	r2_list = []
	model = LinearRegression()
	model.fit(X, y)
	r2_list.append(model.score(X,y))

	while(r2_list[-1] < 1.0):
		X = np.concatenate((X, generate_new_feature()), axis=1)
		model.fit(X, y)
		r2_list.append(model.score(X, y))		

	graph_results(r2_list)


def generate_training_data(): 
	X = np.arange(100)
	noise = np.random.uniform(-10,10,size=(100,))
	y = (.5 * X) + (5 + noise)
	
	return X.reshape(100,1), y.reshape(100,1)

def generate_new_feature():
	return np.random.rand(100).reshape(100,1)

def graph_results(r2):
	x_axis = [i for i in range(len(r2))]
	plt.plot(x_axis, r2)
	plt.xlabel("Number of variables")
	plt.ylabel("R^2")
	plt.show()

main()
