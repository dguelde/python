#Donovan Guelde
#csci 5352 PS5 extra credit
#Yeast transcription network (2002)
#	http://www.weizmann.ac.il/mcb/UriAlon/download/collection-complex-networks for yeast network, 
#US airport networks (2010)
#	http://opsahl.co.uk/tnet/datasets/USairport_2010.txt 

import numpy as np
import random
import time
import math
import matplotlib.pyplot as plt
import igraph

def readFile(fileName):
	with (open(fileName,'r'))as f:
		matrix=np.zeros((1858,1858)) #yeast network has n=688 nodes, airport network has 1574 (labels go up to 1858)
		for line in f:
			line=line.split()
			node,neighbor=int(line[0])-1,int(line[1])-1 #minus 1 to begin at 0
			matrix[node][neighbor]=1
			matrix[neighbor][node]=1#undirected, unweighted
	f.close()
	return matrix

class GraphHelper(object): #some easy, fast igraph help...not really helpful in this program
	def __init__(self,graph):
		self.graph=graph
		self.adjlist=map(set,graph.get_adjlist())
	
def main():
	#FILENAME="yeastInter_st.txt"
	FILENAME="USairport_2010.txt"
	ITERATIONSPERNODE=2000 #iterations on each node
	matrix = readFile(FILENAME) #numpy matrix
	networkSize=len(matrix)
	g = igraph.Graph.Adjacency((matrix>0).tolist())
	c = igraph.mean(g.degree())
	p = 1./c #transmission probability

	epidemicSize=np.zeros(networkSize) #average cascade size per node
	cascadeSize=np.zeros(ITERATIONSPERNODE) #cascade size per run on patient Zero node
	possibleNewInfections=[] #neighbors of contageous nodes
	newInfections=[] #newly infected nodes at a single time t
	for patientZero in xrange(networkSize): #everybody gets a turn...
		print patientZero
		for iteration in xrange(ITERATIONSPERNODE):
			start = time.time()
			immunity=np.random.rand(networkSize) #immunity chance for nodes
			condition=np.zeros(networkSize) #0=susceptible, 1=contageous, 2=infected but not contageous
			condition[patientZero]=1
			newInfection=True
			while(newInfection):
				newInfection=False
				diseaseSpreaders=np.where(condition==1)
				condition[condition==1]=2 #not contageous any more
				try: #will throw error if no neighbors (if patient zero has no edges...)
					exposed=[neighbors[spreader] for spreader in diseaseSpreaders][0]
				except TypeError:
					continue
				exposed=np.intersect1d(exposed,np.where(condition==0)) #remove non-susceptible from list
				if(len(exposed)==0): continue #if no susceptible, finished
				newInfections=np.intersect1d(exposed,exposed[np.where(immunity[np.array(exposed)]<p)]) #cascade spreads as function of p
				condition[newInfections]=1 #contageous
				if newInfections.sum()>0:
					newInfection=True
			cascadeSize[iteration]=len(np.where(condition!=0)[0]) #if contageous or infected, you count as sick
		epidemicSize[patientZero]=np.average(cascadeSize)
	outputFile=FILENAME[:-4]+"_undirected_{}_iterations_results.txt".format(ITERATIONSPERNODE)
	with (open(outputFile,'w'))as f:
		for index in range(0,networkSize):
			winner=np.argmax(epidemicSize) 
			f.write('{} {}\n'.format(winner+1,epidemicSize[winner]))
			epidemicSize[winner]=0
	f.close()		
		#np.savetxt('./q1c/E{}length.txt'.format(E),epidemicLength)
		#np.savetxt('./q1c/E{}size.txt'.format(E),epidemicSize)
		#plotResults(epidemicSize,epidemicLength,pValues,N,E,C)
		#E+=ESTEP #end of e-loop

main()