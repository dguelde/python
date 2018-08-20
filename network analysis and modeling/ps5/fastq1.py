# Donovan Guelde
# CSCI 5352 PS 5

import numpy as np
import random
import matplotlib.pyplot as plt

INPUTSET= 0 #0-toy data set, 1-boardmember set, 2-malaria
numberTrials=10000
fStep=0.02

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
				metadata[counter]=line
				counter+=1
	f.close()
	metadata = metadata.astype(int)
	# build an n x n simple network.  Uses edge weights to signify class of neighbor node
	# ex.  A(i,j) = 2, A(j,i) = 1--> i and j are linked, j is class 2, i is class 1
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
			matrix[node][neighbor]=metadata[neighbor] 
			matrix[neighbor][node]=metadata[node] # undirected
	f.close()
	temp = np.where(np.sum(matrix,axis=1)==0) #delete vertices with no neighbor info (different year, data set, etc.)
	matrix=np.delete(matrix,temp,axis=0) 
	matrix=np.delete(matrix,temp,axis=1)
	metadata=np.delete(metadata,temp) 
	return matrix,metadata

def main():
	
	if(INPUTSET==0):
		networkFile='toyNetwork.txt'
		metadataFile='toyMetadata.txt'
	if(INPUTSET==1):
		networkFile="net1m_2011-08-01.txt"
		metadataFile="data_people.txt"
	if(INPUTSET==2):
		networkFile='HVR_5.txt'
		metadataFile='metadata_CysPoLV.txt'
	associationMatrix,metadata=readFile(networkFile,metadataFile)
	length = len(metadata)
	numberCategories=metadata.max()-metadata.min()+1
	possibleChoices=np.arange(1,numberCategories+1)
	f=.01
	fCounter=0
	resultsOverF=np.zeros(((0.99-f)/fStep)+1) #store accuracy results for each f value
	fValues=np.zeros(((0.99-f)/fStep)+1) # store f values used for replot, if necessary
	while (f < 1.):
		iterationResults=np.zeros((numberTrials)) #results on each iteration
		iterationCounter=0
		for iteration in xrange(numberTrials):
			trainMatrix=np.copy(associationMatrix) #make a copy so we can alter it w/out losing oiginal
			randomLabels=np.random.randint(1,high=numberCategories+1,size=length)
			randomValues = np.random.random(length) #matrix of 'coin flips' to compare against f for our test set
			hiddenNodes=np.where(randomValues>f)
			while (len(hiddenNodes[0])==0): #test set length 0 makes no sense...try again
				randomValues = np.random.random(length) 
				hiddenNodes=np.where(randomValues>f) #we hide the label on these nodes
			predictions=np.zeros(len(hiddenNodes[0])) #make predictions for nodes w/ hidden labels
			trainMatrix[:,hiddenNodes]=0 #set A(i,j) to 0 when j is hidden (can still see A(j,i) to make predictions for node j)
			findMajority=np.zeros((len(hiddenNodes[0]),numberCategories)) #store 'votes' for each vertex in seperate columns
			for index in range(0,numberCategories): 
				findMajority[:,index]=((trainMatrix==index+1).sum(1))[hiddenNodes] #neighbor vote total for each vertex/class
			predictions=np.zeros(len(hiddenNodes[0])) #store predictions
			predictions[np.where(findMajority[:,0]==findMajority[:,1])]=randomLabels[np.where(findMajority[:,0]==findMajority[:,1])] #if tie (or no votes(tie of 0:0))
			#print findMajority,'\n',predictions
			predictions[np.where(predictions==0)]=(np.argmax(findMajority[np.where(predictions==0)],axis=1))+1 #use majority to determine node class
			correct=float(np.sum(predictions==metadata[hiddenNodes]))
			iterationResults[iterationCounter]=correct/len(hiddenNodes[0])
			iterationCounter+=1
			
		resultsOverF[fCounter]=np.average(iterationResults) #average accuracy of iterations over 1 f value
		fValues[fCounter]=f
		f+=fStep
		fCounter+=1
	plt.plot(fValues,resultsOverF)
	plt.xlabel('f')
	plt.ylabel('Accuracy')
	#plt.savefig('./{}{}Iterations.png'.format(networkFile[:-4],numberTrials))
	plt.show()
	#np.savetxt('./{}{}accuracy.txt'.format(networkFile[:-4],numberTrials),resultsOverF)
	#np.savetxt('./{}{}fValues.txt'.format(networkFile[:-4],numberTrials),fValues)
main()