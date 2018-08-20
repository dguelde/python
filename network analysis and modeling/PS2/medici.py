# Donovan Guelde
# CSCI 5352, Fall '16
# Problem 5
# References: networkx documentation, numpy docs

import networkx
import matplotlib.pyplot as plt
import sys
import numpy as np


class Node:
	def __init__(self,number,name):
		self.name = name
		self.number = number
		self.neighbors = []
	def assignNeighbors(self,neighbors):
		for item in neighbors:
			self.neighbors.append(item)

class Network: 
	def __init__(self,n):
		self.nodes = []
		self.n = n

if __name__ == "__main__":
	network = Network(16)
	names = ["Acciaiuoli","Albizzi","Barbadori","Bischeri","Castellani","Ginori","Guadagni","Lamberteschi","Medici",
				"Pazzi","Peruzzi","Pucci","Ridolfi","Salviati","Strozzi","Tornabuoni"]
	neighbors = {0:[8],1:[5,6,8],2:[4,8],3:[6,10,14],4:[2,10,14],5:[1],6:[1,3,7,15],7:[6],8:[0,1,2,12,13,15],
				9:[13],10:[3,4,14],11:[],12:[8,14,15],13:[8,9],14:[3,4,10,12],15:[6,8,12]}
	graph = networkx.from_dict_of_lists(neighbors)

	shortestPaths = [] #an array to hold ALL shortest paths, to avoid the networkx habit of using only 1 shortest path,
				#even if more exist
	for index in range(0,16):
		for index2 in range(0,16):
			
			try:
				shortestPaths.append([p for p in networkx.all_shortest_paths(graph,index,index2)]) 
								#vertex 11 will cause error
			except(networkx.exception.NetworkXNoPath):
				print"" #do nothing for vertex 11, it has no shortest paths except self-loop
	
	

	print "degree centrality"
	degreeCentrality = []
	centrality = networkx.degree_centrality(graph)
	for index in range(0,16):
		indexCentrality = int(centrality[index]*15)
		print str(index)+":",indexCentrality
		degreeCentrality.append(indexCentrality)
	print degreeCentrality

	print "harmonic centrality"
	for index in range(0,16):
		if(index!=11):
			sum=0
			for index2 in range(0,16):
				if(index!=index2 and index !=11 and index2 != 11): #again, don't try for vertex 11
					sum+=networkx.shortest_path_length(graph,index,index2)
			print (1/float(sum))/15
		else:
			print "0"

	print "eigenvector centrality"
	eigenvectorCentrality = networkx.eigenvector_centrality(graph)
	for index in range(0,16):
		print eigenvectorCentrality[index]

#betweenness, didn't use networkx command to allow for multiple shortest paths
	print "betweenness centrality"
	
	for index in range(0,16):
		counter = 0
		counter2=0
		for item in shortestPaths:
			for item2 in item:
				counter2 += 1
				if (index in item2):
					
					counter+=1

		print (float(counter)/float(counter2))/pow(16,2)


	print "configuration model:"

	
	configurationResults = np.zeros(( 16, 100000 ))
	
	for repetition in range (0,100000):
		tempGraph = networkx.configuration_model(degreeCentrality)
		#perform violence
		tempGraph = networkx.Graph(tempGraph) #collapse multi-edges
		tempGraph.remove_edges_from(tempGraph.selfloop_edges()) #eliminate self-loops
		for index in range(0,16):
			sum=0
			for index2 in range(0,16):
				if(index!=index2):
					try:
						sum+=networkx.shortest_path_length(tempGraph,index,index2)
						
					except (networkx.exception.NetworkXNoPath):
						sum+=0
						
			try:
				configurationResults[index][repetition]=(1/float(sum))/15
				
			except (ZeroDivisionError):
				configurationResults[index][repetition]=0
				
	
	percentilesArray = np.zeros((16,3))
	for index in range (0,16):
		percentilesArray[index][0] = np.percentile(configurationResults[index],25)
		percentilesArray[index][1] = np.percentile(configurationResults[index],50)
		percentilesArray[index][2] = np.percentile(configurationResults[index],75)
	print "25"
	for index in range (0,16):
		print percentilesArray[index][0]
	print "50"
	for index in range (0,16):
		print percentilesArray[index][1]
	print "75"
	for index in range (0,16):
		print percentilesArray[index][2]

