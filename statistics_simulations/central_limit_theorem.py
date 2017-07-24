'''
CLT.py
Simulation demonstrating Central Limit Theorem (Ran program with various samples sizes)
Last Modified: 12/23/2016
Modified By: Andrew Roberts
'''

import random
import numpy as np
import matplotlib.pyplot as plt

SIZE_POP = 1000
FRACTION_SAMP = .5

# Create random population data
population = [random.randint(0,1000) for i in np.arange(SIZE_POP)]

# Generate random samples and store means 
sampleMeans = []
for i in np.arange(500):  
	samp = np.random.choice(population, size=SIZE_POP*FRACTION_SAMP, replace=False)
	# mean = int( round(np.mean(samp)) )
	sampleMeans.append(np.mean(samp))

# Graph
bins = [i for i in range(400, 650, 5)]
plt.title('Sample Size n = {} from population N = {}'.format(FRACTION_SAMP*SIZE_POP, SIZE_POP))
plt.xlabel("Sample means")
plt.ylabel("Frequency")
plt.hist(sampleMeans, bins, histtype='bar', rwidth=.8)
plt.show()
