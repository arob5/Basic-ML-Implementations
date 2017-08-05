#
# markov_class.py
# A class defining a Markov Chain and a second class defining a Markov Reward Process that inherits from the first class
# Last Modified: 8/5/2017
# Modified By: Andrew Roberts
#

import numpy as np

class MarkovChain():
	def __init__(self, P, labels):
		try:
			if (P.shape[0] != len(labels)) or (P.shape[1] != len(labels)):
				raise Exception
		except Exception:
			print("Invalid input: Dimensions of transition matrix must match label length")
		else:
			self.P = P
			self.labels = labels
	
	def sample_episode(self, start_state):
		"""
		Returns a single sample episode of the Markov Process
		Arguments: The label of the state to begin in 
		"""
		s = self.labels.index(start_state)
		sample = [start_state]
		state_indices = np.arange(len(self.labels))
		
		while (s != 5):
			next_state_index = np.random.choice(state_indices, p=self.P[s, :])
			sample.append(self.labels[next_state_index])
			s = next_state_index

		return sample

	def state_probability(self, state, start_state,  n=1, n_samples=10000):
		"""
		Returns probability of "state" appearing n times in a sample episode of the Markov process
		Arguments: state - The label of the state of interest
			   start_state - The state to start the sampling from  
			   n     - The number of times state appears in the episode
			   n_itr - Number of samples taken to calulcate probability
		"""
		counter = 0
		for i in np.arange(n_samples):
			if self.sample_episode(start_state).count(state) == n:
				counter += 1
		
		return counter / n_samples


class MRC(MarkovChain):
	def __init__(self, P, labels, rewards):
		super().__init__(P, labels)
		self.values = None
		
		try: 
			if len(rewards) != len(self.labels):
				raise Exception
		except Exception:
			print("Length of reward array must match length of label array")
		else:
			self.rewards = rewards 

	def calc_sample_reward(self, sample):
		""" Returns the total reward of a sample episode passed in as argument"""
		reward_dict = dict(zip(self.labels, self.rewards))

		reward_total = 0
		for label in sample:
			reward_total += reward_dict[label]
		
		return reward_total		

	def store_values(self, discount):
		""" 
		Stores the values of each state as an instance variable
		Value function defined as expected return given a state    	
		Finds values by solving the vectorized Bellman Equation
		Will only work if matrix is invertible 
		
		Argument: A discount factor; 0 <= discount <= 1
		"""

		try:
			if (discount < 0) or (discount > 1):
				raise Exception
		except Exception:
			print("Input Error: Discount value must be between 0 and 1 (inclusive)")
		else: 
			I = np.identity(n=len(self.labels))	
			self.values = np.linalg.inv(I - (discount * self.P)) @ self.rewards.reshape(-1, 1)



states = ["Enroll", "Class", "Drop", "Permanent Drop", "Graduate", "End"]
rewards = [1, 5, -5, -20, 20, 0]
P = np.array([[0, .8, .1, .1, 0, 0], [0, .7, .1, .1, .1, 0], [0, .3, .4, .3, 0, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1]])

mc = MRC(P, states, rewards)
print(mc.store_values(1))



#for i in range(10):
#	print(mc.sample_episode("Enroll"))
