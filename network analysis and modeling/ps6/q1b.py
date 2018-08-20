# Donovan Guelde
# CSCI-5352
# PS6 Q.1.b

import numpy as np
import matplotlib.pyplot as plt
import time



def growNetwork(r,n,pr,c):

	vertexLabelList=np.full(n*c,-1)
	adjacencyList=np.zeros(shape=(n,c))
	adjacencyList.fill(-1) #-1 signifies no edge
	numberVertexes=0
	numberEdges=0
	
	#make a 'seed' graph, c+1 nodes to simulate growth from 0 nodes
	#node 0 has no outgoing edges, node 1 points to 0, node 2 points to 0 and 1, 
	#node 3 points to nodes 0,1,2, etc. Node[c+1] now has c outgoing edges
	for index in range(0,c):
		for index2 in range(index+1,c+1):
			adjacencyList[index2][index]=index
			vertexLabelList[numberEdges]=index
			numberEdges+=1
	numberVertexes=c+1 #we preseeded |c+1| vertexts
	for index in range(numberVertexes,n): #add remaining vertexes to network
		chosen=[]
		tempList=vertexLabelList[0:numberEdges]
		coinFlips=np.random.random(c)
		for index2 in range(0,c): #each new vertex hac c out-degree
			if coinFlips[index2]<pr: #choose in proportion to in-degree
				edgePointsTo=np.random.choice(tempList)
				while (edgePointsTo in chosen or edgePointsTo==-1):
					edgePointsTo=np.random.choice(tempList)
			else: #randomly select
				edgePointsTo=np.random.randint(0,high=numberVertexes-1)
				while(edgePointsTo in chosen):
					edgePointsTo=np.random.randint(0,high=numberVertexes-1)
			chosen.append(edgePointsTo)
		counter=0
		numberVertexes+=1
		for item in chosen: #add new edges to list
			adjacencyList[index][counter]=item
			vertexLabelList[numberEdges]=item
			counter+=1
			numberEdges+=1
	
	return adjacencyList




def main():
	#parameters per assignment
	c=12
	n=1000000
	r=5.
	iterations=100 #?
	fileName = '1b_{}.txt'.format(str(iterations))
	pr=c/(c+r) #probability of attaching to node in proportion of in-degree
	results=np.zeros((iterations,n*.2))
	
	
	
	
	for iteration in xrange(iterations):

		start=time.time()
		adjacencyList=growNetwork(r,n,pr,c).astype(int)
		adjacencyList=adjacencyList[adjacencyList>=0]
		adjacencyList=np.sort(adjacencyList,axis=None)

		for index in xrange(len(adjacencyList)):
			v = adjacencyList[index]

			if ( v < int(n*.1)):
				results[iteration][v]+=1
			
			if (v >= int(n*.9)):
				results[iteration][v-n]+=1		
		
		
		print time.time()-start
	
	average=np.average(results,axis=0)
	np.savetxt(fileName,average,fmt='%1.4f')
	

		

		
main()
