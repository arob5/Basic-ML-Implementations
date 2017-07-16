'''
montyHall.py
Purpose: Running a monty hall simulation
Last Modified: 12/19/2016
Modified By: Andrew Roberts
'''
import random

NUM_ITR = 100000

iterations = 0
correct = 0

while iterations < NUM_ITR:
	# Creates door list and puts prize behind one door (prize = 1)
	doors = [0, 0, 1]
	random.shuffle(doors)

	# Determine location of prize
	prize_loc = 0
	for i in range(0, len(doors)):
		if doors[i] == 1:
			prize_loc = i
			break

	# Randomly choose door 
	choice = random.randint(0,2)	
	if choice == 0:
		if prize_loc == 1:
			switch = 1
		elif prize_loc == 2:
			switch = 2
		else:
			switch = random.randint(1,2)
	elif choice == 1:
		if prize_loc == 2:
			switch = 2
		elif prize_loc == 0:
			switch = 0
		else:
			options = [0,2]
			switch = random.choice(options)
	else:
		if prize_loc == 0:
			switch = 0
		elif prize_loc == 1:
			switch = 1
		else:
			switch = random.randint(0,1)

	# Contestant switches choice to door that was open
	choice = switch

	# Tally correct guesses
	if doors[choice] == 1:
		correct += 1
	iterations += 1

correct_proportion = correct / float(NUM_ITR)
print "Proportion correct: " + `correct_proportion`
