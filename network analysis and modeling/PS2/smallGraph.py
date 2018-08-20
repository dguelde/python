# Donovan Guelde
# CSCI 5352, Fall '16
# Problem 6
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
	names = ["A","B","C","D","E","F","G"]
	neighbors = {0:[1,2],1:[0,2],2:[0,1,3],3:[2,4],4:[3,5],5:[4,6],6:[5]}
	graph = networkx.from_dict_of_lists(neighbors)
	shortestPaths = []
	for index in range(0,7):
		for index2 in range(0,7):
			#if (index != index2 and index != 11 and index2 != 11):
			shortestPaths.append([p for p in networkx.all_shortest_paths(graph,index,index2)])
			
	print "degree centrality"
	degreeCentrality = []
	centrality = networkx.degree_centrality(graph)
	for index in range(0,7):
		indexCentrality = int(centrality[index]*6)
		print str(index)+":",indexCentrality
		degreeCentrality.append(indexCentrality)
	print degreeCentrality
	print "betweenness centrality"
	
	for index in range(0,7):
		counter = 0
		counter2=0
		for item in shortestPaths:
			for item2 in item:
				counter2 += 1
				if (index in item2):
					
					counter+=1

		print (float(counter)/float(counter2))/pow(7,2)