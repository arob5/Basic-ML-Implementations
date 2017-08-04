#
# markov_process.py
# Defines states and a transition matrix for a Markov Process and takes samples of the process
# Last Modified: 8/3/2017
# Modified By: Andrew Roberts
#

import numpy as np

"""
Example of Markov Chain with student rollforward:
    --> s = State Vector
    --> P = Transition Matrix
    --> "End" is an absorbing state

"""

states = ["Enroll", "Class", "Drop", "Permanent Drop", "Graduate", "End"]
P = np.array([[0, .8, .1, .1, 0, 0], [0, .7, .1, .1, .1, 0], [0, .3, .4, .3, 0, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1]])

def find_next_term_status(current_status):
	"""
	Returns student status after one term
	Argument: Vector of current breakdown of student status
	"""
	
	try:
		if current_status.sum() != 1:
			raise Exception
	except Exception:
		print("Elements of vector must sum to 1")
	else:
		return P.T @ current_status

def sample_episode():
	sample = [states[0]]
	s = 0 # Students start at index 0, "Enroll"
	
	while (s != 5):
		next_state_index = np.random.choice([0,1,2,3,4,5], p=P[s, :])
		sample.append(states[next_state_index])
		s = next_state_index
	return sample

to_class = 0
for i in range(1000):
	if sample_episode()[1] == "Class":
		to_class += 1

print("Probability of Enroll->Class = {}".format(to_class/1000))


graduate = 0
for i in range(10000):
	if sample_episode()[-2] == "Graduate":
		graduate += 1

print("Probability of Graduating = {}".format(graduate/10000))

drop = 0
for i in range(10000):
	if sample_episode()[-2] == "Permanent Drop":
		drop += 1

print("Probability of Permanently Dropping = {}".format(drop/10000))

	
'''
current_status = np.array([[.2, .5, .2, .05, .05]]).T
print(current_status)
print(find_next_term_status(current_status))
'''
