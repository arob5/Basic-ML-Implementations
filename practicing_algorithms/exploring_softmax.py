#
# exploring_softmax.py
# Exploring the softmax activation function and plotting the softmax function against the reLU function
# Last Modified: 7/25/2017
#
#
# Softmax activation: P(y=j | zi) = exp(zi) / sigma(exp(zk))
# Softmax function: For x1,...,xk -> ln(sigma(exp(xi)))
#

import matplotlib.pyplot as plt
import numpy as np


def softmax(*args):
	""" Returns approximation of max function using softmax """
	input = np.array([np.exp(arg) for arg in args])
	return np.log(input.sum())

def softmax_activation(*args): 
	""" Takes an array and outputs softmax activation output elementwise"""
	input = np.array([np.exp(arg) for arg in args])
	input_sum = input.sum()
	return [val/input_sum for val in input]

# First, graphing rectified linear unit against softmax()
x = np.arange(-10, 10)
y_reLU = [max(0, i) for i in x]
y_softmax = [softmax(0, i) for i in x]

plt.plot(x, y_reLU, color="blue", label="reLU")
plt.plot(x, y_softmax, color="red", label="Softmax")
plt.ylim([-2,11])
plt.suptitle("ReLU vs. Max, maximizing (0,x)")
plt.xlabel("x")
plt.ylabel("Softmax/reLu(0, x)")
plt.legend(loc="upper left")
plt.show()

# Now observing behavior of softmax activation funtion
test_input = np.random.randint(-10, 10, 3)
print "Input: ", test_input
print "Output: ", softmax_activation(*test_input)
