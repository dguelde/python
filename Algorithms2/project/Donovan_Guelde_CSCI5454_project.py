# Donovan Guelde
# CSCI 5454 Final Project
# implementation of Christofides algorithm

import networkx as nx
import random
import numpy as np
from itertools import permutations
from sys import maxint
from math import factorial
import time


NUMBEROFPOINTS=8 # number of 'cities' for TSP
NUMBERITERATIONS=50
MAPSIZE=1000 #dimension for n x n 'map'



#randomly generate locations (x,y) for NUMBEROFPOINTS locations on a MAPSIZExMAPSIZE plane
#return numpy matrix of (x,y) locations
def makeMap():
	locations=np.zeros((NUMBEROFPOINTS,2))
	counter=0
	for index in range(NUMBEROFPOINTS):
		xCoord=np.random.randint(0,MAPSIZE)
		yCoord=np.random.randint(0,MAPSIZE)
		locations[counter][0]=xCoord
		locations[counter][1]=yCoord
		counter+=1
	return locations

#generate and return a networkX instance of graph
#with weighted edges based on locations
def makeGraph(locations):
	g = nx.Graph()
	for startingPoint in range(NUMBEROFPOINTS):
		for destination in range(startingPoint+1,NUMBEROFPOINTS):
			distance = np.sqrt((locations[startingPoint][0]-locations[destination][0])**2 + (locations[startingPoint][1]-locations[destination][1])**2)
			g.add_edge(startingPoint,destination,weight=distance)
			
	return g

#generate and return a numpy matrix of distances between all 'cities'
def getDistances(locations):
	distances=np.zeros((NUMBEROFPOINTS,NUMBEROFPOINTS))
	for startingPoint in range(NUMBEROFPOINTS):
		for destination in range(NUMBEROFPOINTS):
			distances[startingPoint][destination]=np.sqrt((locations[startingPoint][0]-locations[destination][0])**2 + (locations[startingPoint][1]-locations[destination][1])**2)
	return distances

#generate and return a minimum-weight matching
#networkX uses a maximum-weight matching, so to get minimum-matching,
#subtract distance[a][b] from a large number, generate graph with new distances,
#then find the maximum-matching.  Distances are inversley proportional to original(large->small, small->large)
#but matching reflects a min-match of original locations
def getMinMatching(degreeList,distances):
	maxDistance=np.amax(distances)+1
	tempDistances=np.subtract(maxDistance,distances)
	tempGraph=nx.Graph()
	for origin in degreeList:
		for destination in degreeList:
			tempGraph.add_edge(origin,destination,weight=tempDistances[origin][destination])
			
	minMatch=nx.max_weight_matching(tempGraph)
	return minMatch

#sums up the TSP approximation
def calculateDistance(tspSolution,distances):
	total=0
	#print tspSolution
	for index in range(0,NUMBEROFPOINTS):
		#print index
		total+=distances[tspSolution[index]][tspSolution[index+1]]
	return total

#simple brute-force implementation of TSP
def bruteForceTSP(distances):
	cities=np.arange(1,NUMBEROFPOINTS)
	start=0
	possibleRoutes=permutations(cities,NUMBEROFPOINTS-1)
	minimum=maxint
	for item in possibleRoutes:
		route=list(item)
		if(route[0]<route[-1]):
			total=distances[0][route[0]]
			for index in range(1,NUMBEROFPOINTS-1):
				total+=distances[route[index-1]][route[index]]
				if total>minimum:
					continue
			total+=distances[0][route[index]]
			if total<minimum:
				minimum=total
			
	return minimum



if __name__ == "__main__":
	trueDistance=0.
	approximateDistance=1.
	counter=0
	worstRatio=0
	bestRatio=2
	algTime=0
	bruteForceTime=0
	algTimeArray=np.zeros(NUMBERITERATIONS)
	bruteForceTimeArray=np.zeros(NUMBERITERATIONS)
	algResults=np.zeros(NUMBERITERATIONS)
	bruteForceResults=np.zeros(NUMBERITERATIONS)
	ratio=2
	while(counter<NUMBERITERATIONS):
		locations=makeMap()
		distances=getDistances(locations)
		g = makeGraph(locations)
		mst = nx.minimum_spanning_tree(g)
		degreeList=nx.degree(mst)
		oddDegree=[]
		for index in range(len(degreeList)):
			if degreeList[index]%2==1:
				oddDegree.append(index)
		minMatch=getMinMatching(oddDegree,distances)
		minMatchGraph=nx.Graph()
		for item in minMatch:
			minMatchGraph.add_edge(item,minMatch[item],weight=distances[item][minMatch[item]])
		H=nx.MultiGraph()
		H.add_edges_from(mst.edges()+minMatchGraph.edges())
		path=[]
		for item in list(nx.eulerian_circuit(H)):
			path.append(item[0])
		path.append(item[1])
		tspSolution=[]
		for item in path:
			if item not in tspSolution:
				tspSolution.append(item)
		tspSolution.append(path[-1])
		start=time.time()
		approximateDistance=calculateDistance(tspSolution,distances)
		algTime=time.time()-start
		start2=time.time()
		trueDistance=bruteForceTSP(distances)
		bruteForceTime=time.time()-start2
		ratio=approximateDistance/trueDistance
		
		if ratio>worstRatio:
			worstRatio=ratio
		if ratio<bestRatio:
			bestRatio=ratio
		algTimeArray[counter]=algTime
		bruteForceTimeArray[counter]=bruteForceTime
		print "iteration",counter,"ratio",ratio,"worst ratio",worstRatio,"best ratio",bestRatio,"time ratio",np.sum(algTimeArray)/np.sum(bruteForceTimeArray)
		algResults[counter]=approximateDistance
		bruteForceResults[counter]=trueDistance
		
		counter+=1
	np.savetxt('algResults{}.txt'.format(NUMBEROFPOINTS),algResults)
	np.savetxt('bruteForceResults{}.txt'.format(NUMBEROFPOINTS),bruteForceResults)
	np.savetxt('algTimes{}.txt'.format(NUMBEROFPOINTS),algTimeArray)
	np.savetxt('bruteForceTimes{}.txt'.format(NUMBEROFPOINTS),bruteForceTimeArray)

	



