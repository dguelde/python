# Donovan Guelde
# CSCI 5352 PS 5

import numpy as np
import random
import time
import matplotlib.pyplot as plt

INPUTSET=0 #0-toy data set, 1-boardmember set, 2-malaria
numberTrials=1000
fStep=0.01

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
			matrix[node][neighbor]=metadata[neighbor] # use neighbor's node type as edge weight for easy use via numpy
			matrix[neighbor][node]=metadata[node] # undirected
	f.close()
	matrix = matrix.astype(int)
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
	f=.0
	fCounter=0
	resultsOverF=np.zeros(((1.-f)/fStep))+1 #store accuracy results for each f value
	fValues=np.zeros(((1.-f)/fStep))+1 # store f values used for replot, if necessary
	while (f < 1.0):
		iterationResults=np.zeros((numberTrials)) #results on each iteration
		iterationCounter=0
		start = time.time()
		for iteration in xrange(numberTrials):
			trainMatrix=np.copy(associationMatrix) #make a copy so we can alter it w/out losing oiginal
			predictions=np.zeros(length) 
			randomValues = np.random.random((length)) #matrix of 'coin flips' to compare against f for our test set
			trainMatrix[:,np.where(randomValues>f)]=0 #set A(i,j) to 0 when j is hidden (can still see A(j,i) to make predictions for node j)

			#filteredArrays = np.zeros((numberCategories,len(trainMatrix),len(trainMatrix[0]))) #seperate numpy array for each class
			counter=0 
			for index in range(metadata.min(),metadata.max()+1):
				temp = np.zeros_like(trainMatrix)
				temp[np.where(trainMatrix==index)]=1
				filteredArrays[counter] = temp
				counter+=1


			#filteredArrays has a matrix for every category value, so the index of largest sum
			#gives category with most (or category that is tied for most) votes
			possibleChoices=np.arange(1,numberCategories+1)
			findMajority=np.zeros((length,numberCategories)) #store 'votes' 
			for index in range(0,numberCategories): 
				for index2 in range(0,length):
					findMajority[index2][index]=np.sum(filteredArrays[index][index2]) #neighbor vote total for each class
			predictions=np.zeros(len(findMajority)) #store predictions
			for index in range(0,len(findMajority)):
				if (np.argmax(findMajority[index])!=0): #if there are votes from neighbors
					predictions[index]=np.argmax(findMajority[index])+metadata.min() #index of max value
					temp=predictions[index]
					tie=np.where(findMajority[index]==temp-1) #check for ties
					
					if (len(tie[0])>1): #if tie, break with random.choice()
						
						tiebreaker=np.random.choice(tie[0])
						
						predictions[index]=tiebreaker
				else: # no votes from neighbors, just guess from possible categories
					predictions[index]=np.random.choice(possibleChoices)
			
			correct=0. #find accuracy of iteration
			for index in xrange(length):
				if predictions[index]==metadata[index]:
					correct+=1.
			iterationResults[iterationCounter]=correct/length
			iterationCounter+=1
		print predictions.astype(int)	
		print metadata
		print trainMatrix
		resultsOverF[fCounter]=np.average(iterationResults) #average accuracy of iterations over 1 f value
		fValues[fCounter]=f
		print time.time()-start,f
		f+=fStep
		fCounter+=1
	
	plt.plot(fValues,resultsOverF)
	plt.xlabel('f')
	plt.ylabel('Accuracy')
	plt.savefig('./{}{}Iterations.png'.format(networkFile[:-4],numberTrials))
	plt.show()
	np.savetxt('./{}{}accuracy.txt'.format(networkFile[:-4],numberTrials),resultsOverF)
	np.savetxt('./{}{}fValues.txt'.format(networkFile[:-4],numberTrials),fValues)
	
		






main()