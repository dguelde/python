# Donovan Guelde
# CSCI 5454
# PS3, Q4

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
from math import sqrt
from math import ceil
from sys import maxint

def generateGraph(n,p):
	tempGraph = nx.gnp_random_graph(n,p) #generate random graph (this gives simple graph, not gonna work)
	while(nx.number_connected_components(tempGraph)!=1): #make sure graph is connected...
		tempGraph = nx.gnp_random_graph(n,p)
	graph = nx.MultiGraph() 
	for edge in tempGraph.edges(): #copy edges into multigraph to preserve multi-edges after node collapse
		graph.add_edge(edge[0],edge[1])
	return graph

def karger(g,T):
	graph = g
	counter=0
	while (len(graph.nodes())>T): #repeat until only T nodes remaining
		edge = random.sample(graph.edges(),1)[0] #randomly select edge for collapse; returns an array of length 1, just get [0]
		v1,v2 = edge[0],edge[1] #nodes at edge endpoints, gonna combine these 2
		neighbors = graph.neighbors(v1) #set of v1's neighbors
		v2 = np.random.choice(neighbors) #combine v1,v2
		for edge in graph.edges(v1): #add v1's edges to v2
			if edge[1] != v2: #no self-loops
				graph.add_edge(v2,edge[1])
		graph.remove_node(v1) #remove v1 and its edges from graph
	return graph #returns graph with T nodes remaining (T = )

def kargerStein(g,n):
	g = karger(g,n) #perform karger to limit
	minCut= len(nx.minimum_edge_cut(g))
	return minCut


def main():
	N=[2,3,5,8,13,21]
	results = [] #store results for plotting later
	#n=[5]
	counter=0
	for n in N:
		T = ceil(1+n/sqrt(2))
		print "T=",T
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
				cutWeight=kargerStein(graph,T)
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
	plt.title('Distribution of Accuracy of Karger-Stein Algorithm\nas a Function of Number of Vertices')
	plt.xlabel('Number of Vertices')
	plt.ylabel('Accuracy')
	plt.savefig('kargerStein.png')
	#plt.show()

	#position = nx.spring_layout(graph)
	#labels=nx.draw_networkx_labels(graph,position)
	#nx.draw(graph,position)
	#plt.show()


main()