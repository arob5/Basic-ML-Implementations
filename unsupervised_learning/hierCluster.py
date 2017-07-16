'''
hierCluster.py
Purpose: Practicing hierarchical clustering in Euclidean Space
Last Modified: 12/26/2016
Modified By: Andrew Roberts
'''
import math

def calcDist(index1, index2):
	xDist = ( clusterList[index1].avg[0] - clusterList[index2].avg[0] ) ** 2
	yDist = ( clusterList[index1].avg[1] - clusterList[index2].avg[1] ) ** 2
	return math.sqrt(xDist + yDist) 



class Cluster:
	def __init__(self, vec, left=None, right=None, avg=[0.0,0.0], id=None):
		self.left  = left
		self.right = right
		self.vec   = vec
		self.id    = id
		self.avg  = avg

x = [[1, 3], [2, 2], [4, 2], [10, 10], [9, 12], [12, 9], [20, 22], [21, 21], [22, 20]]

# Populate cluster list
clusterList = [Cluster(vec=[x[i]], id=i, avg=x[i]) for i in range(0, len(x))]

while len(clusterList) > 3:
	# Determine two closest clusters
	closest = (0,1)
	closestDist = calcDist(0, 1)
	for i in range(len(clusterList)):
		for j in range(i+1, len(clusterList)):
			dist = calcDist(i,j)
			if dist < closestDist:
				closest = (i, j)
				closestDist = dist

	# Combine clusters
	clust1 = clusterList[closest[0]]
	clust2 = clusterList[closest[1]]	
	newVec = clust1.vec
	newVec.append(clust2.vec[0])
	newAvg = [(clust1.avg[0] + clust2.avg[0])/float(2), (clust1.avg[1] + clust2.avg[1])/float(2)]
	clusterList.append(Cluster(vec=newVec, left=clust1.vec, right=clust2.vec, avg=newAvg))

	if closest[0] > closest[1]: 
		del clusterList[closest[0]]
		del clusterList[closest[1]]
	else:
		del clusterList[closest[1]]
		del clusterList[closest[0]]
	

# Print 3 Final Clusters
print "Cluster 1"
print clusterList[0].vec

print "Cluster 2"
print clusterList[1].vec

print "Cluster 3"
print clusterList[2].vec
