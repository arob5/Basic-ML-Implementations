'''
CLT.py
Simulation demonstrating Central Limit Theorem (Ran program with various samples sizes)
Last Modified: 12/23/2016
Modified By: Andrew Roberts
'''

import random
import numpy as np
import matplotlib.pyplot as plt

sampSize = 500

# Create random population data
population = []
count = 0
while count < 1000:
	population.append(random.randint(0,1000))
	count += 1

# Generate random samples and store means 
sampleMeans = []
numSamples = 0 
while numSamples < 500:
	samp = []
	samp = np.random.choice(population, size=sampSize, replace=False)
	mean = int( round(np.mean(samp)) )
	sampleMeans.append(mean)
	numSamples += 1

# Graph
bins = []
binValue = 400
while binValue <= 650:
	bins.append(binValue)
	binValue += 5

plt.title('Sample Size n = 500')
plt.hist(sampleMeans, bins, histtype='bar', rwidth=.8)
plt.show()
