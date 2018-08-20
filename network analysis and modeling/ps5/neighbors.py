import numpy as np

INPUTSET=0
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
			matrix[node][neighbor]=1 # use neighbor's node type as edge weight for easy use via numpy
			matrix[neighbor][node]=1 # undirected
	f.close()
	matrix = matrix.astype(int)
	temp = np.where(np.sum(matrix,axis=1)==0) #delete vertices with no neighbor info (different year, data set, etc.)
	matrix=np.delete(matrix,temp,axis=0) 
	matrix=np.delete(matrix,temp,axis=1)
	metadata=np.delete(metadata,temp) 
	return matrix,metadata


def main():
	networkFile='toyNetwork.txt'
	metadataFile='toyMetadata.txt'
	matrix,metadata=readFile(networkFile,metadataFile)
	print matrix
	print "\n"
	print np.dot(matrix,matrix)
	


main()