#
# markov_process.py
# Defines states and a transition matrix for a Markov Process and takes samples of the process
# Last Modified: 8/1/2017
# Modified By: Andrew Roberts
#

import numpy as np

"""
Example of Markov Chain with student rollforward:
    --> s = State Vector
    --> P = Transition Matrix
    --> Permanent Drop and Graduate are absorbing states

"""


states = ["Enroll", "Class", "Drop", "Permanent Drop", "Graduate"]
P = np.array([[0, .9, .1, .1, 0], [0, .7, .1, .1, .1], [0, .3, .4, .3, 0], [0,0,0,0,0], [0,0,0,0,0]])

#
# Exercise 1: Change of student status after 1 term
#

current_status = np.array([.2, .5, .05, .2, .05])
next_term_status = P.T @ current_status
print(next_term_status)
