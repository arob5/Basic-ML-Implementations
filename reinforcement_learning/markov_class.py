#
# markov_class.py
# A class defining a Markov Chain and a second class defining a Markov Reward Process that inherits from the first class
# Last Modified: 8/3/2017
# Modified By: Andrew Roberts
#

import numpy as np

class MarkovChain():
	def __init__(self, P, labels):
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
		self.rewards = rewards

	def calc_sample_reward(self, sample):
		reward_dict = dict(zip(self.labels, self.rewards))

		reward_total = 0
		for label in sample:
			reward_total += reward_dict[label]
		
		return reward_total		




"""
states = ["Enroll", "Class", "Drop", "Permanent Drop", "Graduate", "End"]
P = np.array([[0, .8, .1, .1, 0, 0], [0, .7, .1, .1, .1, 0], [0, .3, .4, .3, 0, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1]])

mc = MarkovChain(P, states)

for i in range(10):
	print(mc.sample_episode("Enroll"))
"""
