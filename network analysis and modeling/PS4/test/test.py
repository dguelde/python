import networkx as nx
import numpy as np
import random
import time
import math
import matplotlib.pyplot as plt

"""
main() consists of 3 nested loops for the 'easter egg hunt',
outer loop is an iteration over a range of epsilon values
middle loop is an iteration over p values
inner loop is a loop over a specific graph instance
"""
def infectThePopulation(neighborMatrix,p,N):
	#print "\n"
	print neighborMatrix
	random.seed(time.time())
	susceptible = np.ones(N) #all susceptible
	contageous=np.zeros(N) #no contageous
	infected=np.zeros(N) #no infected
	patientZero=int(N*random.random())
	#print patientZero
	susceptible[patientZero]=0
	infected[patientZero]=1
	contageous[patientZero]=1
	newInfection=1
	t=0
	while (newInfection==1): #while disease spreads (if doesn't spread, then no new contageous nodes to check)
		#newInfection=0
		#newInfections=[]
		spreaders = np.where(contageous==1)
		print spreaders
		for person in spreaders[0]:
			print neighborMatrix[person]
			if susceptible[np.where(neighborMatrix[person]==1)]==1:
				immunity=random.random()
				if immunity<p:
					newInfection=1
					#print "sick",victim
					newInfections.append(victim)


			#contageous[np.where(contageous==1)]=0
			contageous[person]=0 #not contageous any more
			#print person,contageous[person]
			for victim2 in newInfections:
				infected[victim2]=1
				contageous[victim2]=1
				susceptible[victim2]=0
				newInfections.remove(victim2)		
		t=t+1
	

	size = np.count_nonzero(infected)
	return (size,t)

def plotResults(epidemicSize,epidemicLength,pValues,N,E,C): #make some graphs...
	plt.plot(pValues,epidemicSize)
	plt.title('Average Epidemic Size\n(n={}, C={}, Epsilon={})'.format(N,int(C),E))
	plt.xlabel('P')
	plt.ylabel('Epidemic Size')
	
	plt.savefig('./E{}size.png'.format(E))
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
	
	plt.axhline(y=math.log(N),xmin=0,xmax=1,color='r',ls='dashed')
	plt.savefig('./{}Elength.png'.format(E))
	plt.close()

def main():
	#important variables
	ITERATIONSONP=2 # number of iterations for each p value
	ITERATIONSPERGRAPH=2 #iterations on each graph
	Emin=0.0 #range of epsilon values to consider
	Emax=15.9
	ESTEP=.2
	PMIN=0
	PMAX=1#range of p values to consider
	PSTEP=.1
	C=8
	N=200
	L=2
	

	size=((PMAX-PMIN)/PSTEP)+1
	epidemicSize=np.zeros((size)) #hold results from outer loop
	epidemicLength=np.zeros((size))
	pValues=np.zeros((size))
	
	E=Emin
	while (E < Emax): #iterate on a range of epsilon values
		c=float(C)
		k=int(N/L) #k=vertices per group
		c_in=2*C+E
		c_out=2*C-E
		p_in=(.5*c_in)/N
		p_out=(.5*c_out)/N
		p=PMIN
		counter=0

		while (p<PMAX): #next inner loop, over p values
			start = time.time()
			pValues[counter]=p #use this p on multiple generated graphs (multiple times)
			sizeArray=np.zeros((ITERATIONSONP)) #store size results for runs on multiple graphs
			lengthArray=np.zeros((ITERATIONSONP)) #store length results for runs on multiple graphs
			for index in range(0,ITERATIONSONP): 
				graphInfectionSize=np.zeros((ITERATIONSPERGRAPH)) #store size results for multiple infections on one graph
				graphInfectionLength=np.zeros((ITERATIONSPERGRAPH)) #store length results for multiple infections on one graph
				g = nx.planted_partition_graph(L,k,p_in,p_out) #generate planted partition graph
				AssociationMatrix = nx.to_numpy_matrix(g)
				neighbors=[]
				counter2=0
				for item in g:
					neighbors.append(nx.neighbors(g,counter2))
					counter2+=1
				neighborMatrix = np.asarray(neighbors)
				for index2 in range(0,ITERATIONSPERGRAPH): #iterate on a graph
					graphInfectionSize[index2],graphInfectionLength[index2]=infectThePopulation(neighborMatrix,p,N) #run scenario on graph
				sizeArray[index]=np.sum(graphInfectionSize)/ITERATIONSPERGRAPH #average size of infection from the given graph
				lengthArray[index]=(np.sum(graphInfectionLength))/ITERATIONSPERGRAPH #average length from given graph
			epidemicLength[counter]=(np.sum(lengthArray))/ITERATIONSONP #average infection length from multiple graphs for given value of p
			epidemicSize[counter]=np.sum(sizeArray)/ITERATIONSONP	#average infection size from multiple graphs for given value of p
			p+=PSTEP #end of p-loop
			counter+=1
			print "E",E,"p",p,time.time()-start
		np.savetxt('./length.txt'.format(E),epidemicLength)
		np.savetxt('./size.txt'.format(E),epidemicSize)
		plotResults(epidemicSize,epidemicLength,pValues,N,E,C)
		E+=ESTEP #end of e-loop

main()