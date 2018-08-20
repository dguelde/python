# Donovan Guelde
# CSCI 5354 PS4
# Fall 2016

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import time


###globals(bad...), for easy achanges to scenario, # of repetitions, etc

###





def showGraph(graph): #look at the pretty pictures...
	position = nx.spring_layout(graph)
	labels=nx.draw_networkx_labels(graph,position)
	nx.draw(graph,position)
	plt.show()
	return

def spreadDisease(graph,p): #returns 1 if cascade spread, 0 if not, and updates node attributes to reflect changes
	newInfection=0
	newInfections=[]
	appendNewInfections=newInfections.append
	for node in graph: #for each node
		if graph.node[node]['C']==2: #if it is contageous
			for neighbor in nx.neighbors(graph,node):
				if graph.node[neighbor]['C']==1: # if its neighbor is susceptible
					#print node,neighbor
					didYouWashYourHands=random.random()
					if didYouWashYourHands<p: # flip the coin
						#print "infected"
						newInfection=1
						#print newInfection
						appendNewInfections(neighbor)
		#print newInfections
			graph.node[node]['C']=3 #this node not contageous any more
	#update new infections
	for node in newInfections:
		#graph.node[node]['I']=1 #infected
		graph.node[node]['C']=2 #condition 1=susceptible, 2=contageous,3=infected
		#print node,"infected"
		#graph.node[node]['S']=0 #not susceptible
	#print newInfection
	return newInfection

def infectThePopulation(graph,p): #run scenario for given graph,p
	patientZero=int(len(graph.nodes())*random.random())
	#graph.node[patientZero]['I']=1 #infected
	graph.node[patientZero]['C']=2 #contageous
	#graph.node[patientZero]['S']=0 #not susceptible
	t=1 #start at t=1
	while (spreadDisease(graph,p)==1): #while disease spreads (if doesn't spread, then no new contageous nodes to check)
		#print "round ",t
		t=t+1
	t=t+1 #plus one more for last failed attempt (no new infections)

	size = getTotalInfected(graph)
	return (size,t)

def getTotalInfected(graph): #count nodes with attribute'I'==1
	infected=0
	for node in graph:
		if graph.node[node]['C']==3:
			infected+=1
	return infected

def plotResults(epidemicSize,epidemicLength,pValues,N,E,C): #make some graphs...
	plt.plot(pValues,epidemicSize)
	plt.title('Average Epidemic Size\n(n={}, C={}, Epsilon={})'.format(N,int(C),E))
	plt.xlabel('P')
	plt.ylabel('Epidemic Size')
	plt.axis([0,1,0,N+10])
	plt.savefig('./eggHunt/{}size.png'.format(E))
	plt.close()


	tMax=0
	maxIndex=0
	counter=0.
	for item in epidemicLength:
		if item>tMax:
			tMax=item
			maxIndex=counter
		counter+=1
	maxIndex = float(maxIndex)/100.
	plt.plot(pValues,epidemicLength)
	plt.annotate('Maximum Length {} at p={}'.format(tMax,maxIndex),xy=(maxIndex,tMax))
	plt.title('Average Epidemic Length\n(n={}, C={}, Epsilon={})'.format(N,int(C),E))
	plt.xlabel('P')
	plt.ylabel('Epidemic Length')
	plt.axis([0,1,0,tMax*1.1])
	plt.axhline(y=math.log(N),xmin=0,xmax=1,color='r',ls='dashed')
	plt.savefig('./eggHunt/{}length.png'.format(E))
	plt.close()


def main():
	C=8
	E=0
	N=200
	L=2
	PMAX=0.3
	PMIN=0.1
	PSTEP=.001
	ITERATIONSONP=30 # number of iterations for each p value
	ITERATIONSPERGRAPH=30 #iterations on each graph
	Emin=12
	Emax=17

	size=((PMAX-PMIN)/PSTEP)+1
	epidemicSize=np.zeros((size))
	epidemicLength=np.zeros((size))
	pValues=np.zeros((size))
	
	E=Emin
	while (E < Emax):
		c=float(C)
		k=int(N/L) #k=vertices per group
		c_in=2*C+E
		c_out=2*C-E
		p_in=(.5*c_in)/N
		p_out=(.5*c_out)/N
		p=PMIN
		counter=0
		while (p<PMAX): #for all p = [0,1]
			print p
			start = time.time()
			pValues[counter]=p #use this p on multiple generated graphs (multiple times)
			sizeArray=np.zeros((ITERATIONSONP)) #store size results for runs on multiple graphs
			lengthArray=np.zeros((ITERATIONSONP)) #store length results for runs on multiple graphs
			for index in range(0,ITERATIONSONP): #iterate on a p value
				graphInfectionSize=np.zeros((ITERATIONSPERGRAPH)) #store size results for multiple infections on one graph
				graphInfectionLength=np.zeros((ITERATIONSPERGRAPH)) #store length results for multiple infections on one graph
				graph = nx.planted_partition_graph(L,k,p_in,p_out) #run simulation on this graph multiple times
				for node in graph:
					graph.node[node]['C']=1 #no contageous nodes at start, all susceptible
				for index2 in range(0,ITERATIONSPERGRAPH): #iterate on a graph
					tempGraph = graph.copy() #make a copy, or iterating on the same graph does nothing after first time...
					graphInfectionSize[index2],graphInfectionLength[index2]=infectThePopulation(tempGraph,p) #run scenario on copy
				sizeArray[index]=np.sum(graphInfectionSize)/ITERATIONSPERGRAPH #average size of infection from the given graph
				lengthArray[index]=(np.sum(graphInfectionLength))/ITERATIONSPERGRAPH #average length from given graph
			epidemicLength[counter]=(np.sum(lengthArray))/ITERATIONSONP #average infection length from multiple graphs for given value of p
			epidemicSize[counter]=np.sum(sizeArray)/ITERATIONSONP	#average infection size from multiple graphs for given value of p
			p+=PSTEP
			counter+=1
			print time.time()-start
		np.savetxt('./eggHunt/{}length.txt'.format(E),epidemicLength,fmt='%d')
		np.savetxt('./eggHunt/{}size.txt'.format(E),epidemicSize,fmt='%d')
		plotResults(epidemicSize,epidemicLength,pValues,N,E,C)
		E+=.1
		


main()