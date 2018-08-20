# Donovan Guelde
# CSCI 5454
# PS3, Q4

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
from sys import maxint

def generateGraph(n,p):
	tempGraph = nx.gnp_random_graph(n,p)
	while(nx.number_connected_components(tempGraph)!=1): #make sure graph is connected...
		tempGraph = nx.gnp_random_graph(n,p)
	graph = nx.MultiGraph()
	for edge in tempGraph.edges():
		graph.add_edge(edge[0],edge[1])
	return graph

def karger(g):
	graph = g
	counter=0
	while (len(graph.nodes())>2):
		edge = random.sample(graph.edges(),1)[0] #randomly select edge for collapse; returns an array, just get [0](its the only one...)
		v1,v2 = edge[0],edge[1] #nodes at edge endpoint, gonna combine these 2
		neighbors = graph.neighbors(v1) #set of v1's neighbors
		v2 = np.random.choice(neighbors) #combine v1,v2
		for edge in graph.edges(v1):
			if edge[1] != v2:
				graph.add_edge(v2,edge[1])
		graph.remove_node(v1)
	return len(graph.edges())

def main():
	N=[2,3,5,8,13,21]
	results = [] #store results for plotting later
	#n=[5]
	counter=0
	for n in N:
		results.append([])
		for index in range(0,100):
			p=random.random()
			while(p<.1):
				p=random.random()
			minCut=maxint
			iterations = 10*pow(n,2)
			minCutCount=0.0
			for index in range(0,iterations):
				graph = generateGraph(n,p)
				cutWeight=karger(graph)
				if cutWeight==minCut:
					minCutCount+=1
				if cutWeight<minCut:
					minCut=cutWeight
					minCutCount=1
			results[counter].append(minCutCount/float(iterations))
			print "n=",n,"p=",p,"weight=",minCut,"count=",minCutCount,"iterations=",iterations,"probability=",minCutCount/float(iterations)
		counter+=1

	plt.boxplot(results)
	plt.xticks([1,2,3,4,5,6],[2,3,5,8,13,21])
	plt.ylim(0,1.05)
	plt.title('Distribution of Accuracy of Karger Algorithm\nas a Function of Number of Vertices')
	plt.xlabel('Number of Vertices')
	plt.ylabel('Accuracy')
	#plt.show()

	plt.savefig('karger.png')

main()