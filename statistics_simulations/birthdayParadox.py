#
# birthdayParadox.py
# A simple simulation looking exploringthe birthday probability paradox
# Modified by: Andrew Roberts
# Last modified: 7/27/2017
#

import matplotlib.pyplot as plt
import scipy.misc as sci
import numpy as np
import random

##############
#            #
# Simulation #
#            # 
##############

def found_match(birthday_list, n_people):
	""" Returns true if there are 2 identical elements in random sample of list, else False"""
	if n_people > len(birthday_list):
		raise ValueError("n_people must be <= length of list")

	birthday_list = np.random.choice(birthday_list, size=n_people, replace=False)

	for i in range(n_people):
		if i == (n_people - 1):
			break
		for person in birthday_list[i+1:]:
			if birthday_list[i] == person:
				return True
	return False



# Generate random birthdays for 100 people
birthdays = [(str(random.randint(1,12)) + "/" + str(random.randint(1, 31))) for i in range(100)]

# Run simulation over various values of n (number of people)
n = 23
NUM_ITR = 100000


print "Simulation results: "

match_count = 0
for i in xrange(NUM_ITR):
	if found_match(birthdays, n):
		match_count += 1

print "Probability of match with {} people in room is {}".format(n, match_count/float(NUM_ITR))	

##############
#            #
# Equation   #
#            # 
##############

def prob_match(n):
	""" Returns probability of birthday match among n people """
	return 1 - (364.0/365.0)**(sci.comb(n,2))

x = range(1,100)
y = [prob_match(num) for num in range(1, 100)]

plt.plot(x, y)
plt.xlabel("Number of People")
plt.ylabel("Probability of Birthday Match")
plt.suptitle("Probability of matching birthday among different size groups")
ax = plt.gca()
ax.grid(True)
plt.show()
