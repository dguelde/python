# Donovan Guelde
# CSCI 5352 PS 5

import numpy as np
import matplotlib.pyplot as plt
import igraph
from sklearn.metrics import roc_auc_score as auc
import time

INPUTSET=0 #0-toy data set, 1-boardmember set, 2-malaria
numberTrials=10
fStep=0.02

#use igraph to quickly(relatively) calculate common neighbor score
#uses association matrix to make igraph instance,
#precalculates adjacency lists to find common neighbor score
#http://stackoverflow.com/questions/28352211/eficient-common-neighbors-and-preferential-attachment-using-igraph
class GraphCalculations(object):
	def __init__(self,graph):
		self.graph=graph
		self.adjlist=map(set,graph.get_adjlist())
	def common_neighbors(self,i,j):
		return np.divide(float(len(self.adjlist[i].intersection(self.adjlist[j]))),len(self.adjlist[i].union(self.adjlist[j])))

#############################################################################################
# Some helper functions to speed things up using triangular matrices rather than full n x n #
#############################################################################################

#input upper triangle array, returns symmetric array
def makeSymmetricFromTriangle(array):
	array = np.add(array,array.T) - np.diag(array.diagonal())
	return array

#returns symmetric array based on average of A[i,j] and A[j,i]
def makeSymmetric(array):
	array=(array+array.T)/2
	return array

#takes in an array, returns upper triangle as a list
def getTriangleMatrixAsList(array):
	arrayList = array[np.triu_indices_from(array)].tolist()
	return arrayList

def readFile(fileName,mData):
	with (open(mData,'r')) as f:#get metadata
		if (INPUTSET!=2): 
			f.next() #skip label row of boardmember,toy set data
		maxNode=0 #number of nodes in network
		for line in f:
			maxNode+=1
		f.seek(0,0)
		metadata = np.zeros((maxNode))
		if (INPUTSET!=2):
			f.next()
		counter=0
		for line in f:
			if (INPUTSET!=2):
				node,gender=line.split()[0],line.split()[-1:][0]
				node=int(node)-1 #index from 0, not 1
				gender=int(gender)
				metadata[node]=gender
			else:
				metadata[counter]=int(line)
				counter+=1
	f.close()

	metadata = metadata.astype(int)
	# build an n x n simple network. 
	with (open(fileName,'r')) as f: 
		lines = f.readlines()
		matrix = np.zeros((maxNode,maxNode)) 
		for line in lines:
			if (INPUTSET!=2):
				node,neighbor = map(int,line.split())
			else:
				node,neighbor = map(int,line.split(','))
			node-=1 #start at [0], not [1]
			neighbor-=1
			matrix[node][neighbor]=1 
			matrix[neighbor][node]=1 # undirected
	f.close()
	#matrix = matrix.astype(int)
	temp = np.where(np.sum(matrix,axis=1)==0) #delete vertices with no neighbor info (different year, data set, etc.)
	matrix=np.delete(matrix,temp,axis=0) 
	matrix=np.delete(matrix,temp,axis=1)
	metadata=np.delete(metadata,temp) 
	
	#matrix=np.ascontiguousarray(matrix)
	metadata=np.ascontiguousarray(metadata)
	return matrix,metadata

def main():
	np.set_printoptions(linewidth=140)
	if(INPUTSET==0):
		networkFile='toyNetwork.txt'
		metadataFile='toyMetadata.txt'
	if(INPUTSET==1):
		networkFile="net1m_2011-08-01.txt"
		metadataFile="data_people.txt"
	if(INPUTSET==2):
		networkFile='HVR_5.txt'
		metadataFile='metadata_CysPoLV.txt'
	matrix,metadata=readFile(networkFile,metadataFile)
	length = len(metadata)
	shape=length,length
	numberCategories=metadata.max()-metadata.min()+1
	f=.01
	fCounter=0
	degreeProductAccuracyOverF=np.zeros(((.99-f)/fStep)+1) #store accuracy results for each f value
	commonNeighborAccuracyOverF=np.zeros(((.99-f)/fStep)+1)
	shortestPathAccuracyOverF=np.zeros(((.99-f)/fStep)+1)
	fValues=np.zeros(((.99-f)/fStep)+1) # store f values used for replot, if necessary
	trueLabels=getTriangleMatrixAsList(matrix) #true edge set in list format

	while (f <= 1.0):
		start = time.time()
		degreeProductIterationResults=np.zeros((numberTrials)) #results on each iteration
		commonNeighborIterationResults=np.zeros((numberTrials))
		shortestPathIterationResults=np.zeros((numberTrials))
		iterationCounter=0
		#start = time.time()
		commonNeighbors=np.empty((length,length))
		for iteration in xrange(numberTrials):
			#determine holdout, generate tie-breaking noise, hide edges
			associationMatrix=np.copy(matrix) #copy original network
			randomValues = np.random.random((length,length)) #matrix of 'coin flips' to compare against f for our test set
			hiddenEdges=np.where(randomValues>f) #flip coin
			hiddenEdgeList=associationMatrix[hiddenEdges]
			associationMatrix[hiddenEdges]=0 #hide edges
			associationMatrix=makeSymmetricFromTriangle(associationMatrix) #use upper triange (after coin flips) to make association matrix
			randomNoise=np.divide(np.random.random((length,length)),length)
			
			#generate degree product scores
			degreeList=np.sum(associationMatrix,axis=1)
			degreeProduct=np.add(np.outer(degreeList,degreeList),randomNoise) #degree product matrix with noise added to break ties
			degreeProductScores=getTriangleMatrixAsList(degreeProduct)

			#generate normalized common neighbor score
			g = igraph.Graph.Adjacency((associationMatrix.astype(bool)).tolist())
			neighborStruct=GraphCalculations(g)
			for index in range(0,length): #make upper triangle matrix via loops
				for index2 in range(index,length):
					commonNeighbors[index][index2]=neighborStruct.common_neighbors(index,index2)
			commonNeighbors=np.nan_to_num(commonNeighbors)
			commonNeighbors=np.add(commonNeighbors,randomNoise)
			commonNeighborScores=getTriangleMatrixAsList(commonNeighbors)

			#generate shortest path score
			shortestPath=np.asarray(g.shortest_paths_dijkstra())
			shortestPath=np.add(shortestPath,randomNoise)
			shortestPath=np.reciprocal(shortestPath) #nodes with no path will have pathlength = (1/noise) -> very large
			shortestPathScores=getTriangleMatrixAsList(shortestPath)

			#get/store AUC scores for iteration
			degreeProductResults=auc(trueLabels,degreeProductScores)
			commonNeighborResults=auc(trueLabels,commonNeighborScores)
			shortestPathResults=auc(trueLabels,shortestPathScores)
			degreeProductIterationResults[iterationCounter]=degreeProductResults
			commonNeighborIterationResults[iterationCounter]=commonNeighborResults
			shortestPathIterationResults[iterationCounter]=shortestPathResults
			iterationCounter+=1

		#accuracy results for each f value
		degreeProductAccuracyOverF[fCounter]=np.average(degreeProductIterationResults) 
		commonNeighborAccuracyOverF[fCounter]=np.average(commonNeighborIterationResults) 
		shortestPathAccuracyOverF[fCounter]=np.average(shortestPathIterationResults)
		fValues[fCounter]=f
		print f,time.time()-start
		f+=fStep
		fCounter+=1
	

	
	plt.plot(fValues,degreeProductAccuracyOverF)
	plt.plot(fValues,commonNeighborAccuracyOverF)
	plt.plot(fValues,shortestPathAccuracyOverF)
	plt.legend(['Degree Product','Common Neighbor','Shortest Path'],loc=4)
	plt.xlabel('f')
	plt.ylabel('AUC')
	#plt.savefig('./predictEdges{}{}Iterations.png'.format(networkFile[:-4],numberTrials))
	plt.show()

	#np.savetxt('./degreeProduct{}{}accuracy.txt'.format(networkFile[:-4],numberTrials),degreeProductAccuracyOverF)
	#np.savetxt('./commonNeighbors{}{}accuracy.txt'.format(networkFile[:-4],numberTrials),commonNeighborAccuracyOverF)
	#np.savetxt('./shortestPath{}{}accuracy.txt'.format(networkFile[:-4],numberTrials),shortestPathAccuracyOverF)
	#np.savetxt('./predictEdges{}{}fValues.txt'.format(networkFile[:-4],numberTrials),fValues)
	

main()
