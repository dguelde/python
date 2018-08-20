# Donovan Guelde
# CSCI-5352
# PS6 Q.1.e

import numpy as np
import matplotlib.pyplot as plt



def growNetwork(r,n,pr,c):

	vertexLabelList=np.full(n*c,-1)
	adjacencyList=np.zeros(shape=(n,c))
	adjacencyList.fill(-1)
	numberVertexes=0
	numberEdges=0
	iterations=1
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
		coinFlips=np.random.random(c)
		tempList=vertexLabelList[0:numberEdges]
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
	return adjacencyList[1:] 


def growNetworkNonpreferential(n,c):
	vertexLabelList=np.full(n*c,-1)
	adjacencyList=np.zeros(shape=(n,c))
	adjacencyList.fill(-1)
	numberVertexes=0
	numberEdges=0
	iterations=1
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
		for index2 in range(0,c): #each new vertex hac c out-degree
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
	return adjacencyList[1:] 


def main():
	#parameters per assignment
	c=3
	N=1000000
	XaxisResults=[]
	YaxisResults=[]

	R=[1.,4.]
	#bigger r --> lower pr --> higher chance of random assignment of new edges --> more uniform distribution
	for r in R:
		pr=c/(c+r) #probability of attaching to node in proportion of in-degree
		n=N
		fileName = '1e_plot_r={}.png'.format(str(r))

		adjacencyList=growNetwork(r,n,pr,c)
		nodes,indegree=np.unique(adjacencyList,return_counts=True) #
		uniqueValues=np.unique(indegree)
		indegreeList=np.zeros(n)
		counter=0
		for item in nodes:
			if(item!=-1):
				indegreeList[item]=indegree[counter]
			counter+=1
		
		xAxis=np.unique(indegreeList) #observed degree on x axis
		XaxisResults.append(xAxis)
		yAxis=np.zeros(len(xAxis))
		counter=0
		for item in uniqueValues:
			yAxis[counter]=np.count_nonzero(indegreeList>=item) #fraction (nodes indegree >= indegree k)
			counter+=1
		yAxis=yAxis/float(np.amax(yAxis)) #regularize as a fraction < 1
		YaxisResults.append(yAxis)
		
	adjacencyList=growNetworkNonpreferential(n,c) #non-preferential attachment graph
	nodes,indegree=np.unique(adjacencyList,return_counts=True) #
	uniqueValues=np.unique(indegree)
	indegreeList=np.zeros(n)
	counter=0
	for item in nodes:
		if(item!=-1):
			indegreeList[item]=indegree[counter]
		counter+=1
	
	xAxis=np.unique(indegreeList) #observed degree on x axis
	XaxisResults.append(xAxis)
	yAxis=np.zeros(len(xAxis))
	counter=0
	for item in uniqueValues:
		yAxis[counter]=np.count_nonzero(indegreeList>=item) #fraction (nodes indegree >= indegree k)
		counter+=1
	yAxis=yAxis/float(np.amax(yAxis)) #regularize as a fraction < 1
	YaxisResults.append(yAxis)

	labels=['Price\'s Model, R=1','Price\'s Model, R=4','Non-preferential Growth Model']
	plt.xlabel('in-degree k_in')
	plt.ylabel('Pr(K>k_in)')
	plt.xlim(1,n)
	for index in range(0,len(R)+1):
		plt.loglog(XaxisResults[index],YaxisResults[index],label=labels[index])
	
	plt.legend(loc='upper right')
	plt.savefig(fileName)
	plt.close()
	
main()
