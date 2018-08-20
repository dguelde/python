# Donovan Guelde
# CSCI 5352
# EC #6
# references: networkx online documentation, https://docs.python.org/2/library/functions.html#max

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random

DOPART1=1
DOPART2=1
#calculate mean fractional size per part 1, Q6
if (DOPART1):
	NUMBERITERATIONS=2663 #confidence level-99, confidence interval-2.5
	resultsArray = np.zeros(NUMBERITERATIONS)
	for index in range(0,NUMBERITERATIONS):
		degreeSequence=[0]*1000
		degreeSequence[0]=1
		while (not nx.is_valid_degree_sequence(degreeSequence)):
			for index2 in range (0,1000): #generate degree sequence
				rollTheDice=random.random()
				if (rollTheDice<=.6): 
					degreeSequence[index2]=1
				else: 
					degreeSequence[index2]=3
		graph = nx.configuration_model(degreeSequence)
		resultsArray[index] = len(max(nx.connected_component_subgraphs(graph),key=len))
	print "mean of largest component after",NUMBERITERATIONS,"iterations =",np.mean(resultsArray)
	print "mean fractional size =",np.mean(resultsArray)/1000



if (DOPART2):
	NUMBERITERATIONS=666 #confidence level-99, confindence interval-5
	resultsArray = np.zeros(100)
	p1Values = np.zeros(100)
	p1=.01
	counter=0
	while (p1<=1.01):
		iterationResultsArray = np.zeros(NUMBERITERATIONS)
		for index in range(0,NUMBERITERATIONS):
			degreeSequence=[0]*1000
			degreeSequence[0]=1
			while (not nx.is_valid_degree_sequence(degreeSequence)):
				for index2 in range (0,1000): #generate degree sequence
					rollTheDice=random.random()
					if (rollTheDice<=p1): 
						degreeSequence[index2]=1
					else: 
						degreeSequence[index2]=3
			graph = nx.configuration_model(degreeSequence)
			iterationResultsArray[index] = len(max(nx.connected_component_subgraphs(graph),key=len))
		resultsArray[counter]=np.mean(iterationResultsArray)/1000
		p1Values[counter]=p1
		p1+=.01
		counter+=1
	
	plt.ylabel('mean fractional size of largest component')
	plt.xlabel('p1')
	plt.title('Mean Fractional Size of Largest Component as a Function of p1')
	plt.xlim(.01,1)
	plt.plot(p1Values,resultsArray)
	plt.savefig("meanFractionalSize.png")
	plt.close()
	



	




