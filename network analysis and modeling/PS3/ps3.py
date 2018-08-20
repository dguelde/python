# Author: Donovan Guelde
# CSCI 5352 PS3 Question 3
# references: online documentation for numpy, 
#		http://stackoverflow.com/questions/9754729/remove-object-from-a-list-of-objects-in-python
# Collaborators: None

import numpy as np
import os
import matplotlib.pyplot as plt
import networkx as nx

#fileName="test.txt"
fileName="../../CSCI5352_Data/karate_club_edges.txt"

def inModGroup(i,maxModularityGroups):
	if i in maxModularityGroups[0]:
		return 0
	if i in maxModularityGroups[1]:
		return 1
	if i in maxModularityGroups[2]:
		return 2

class Network:
	def __init__(self,fileName):
		self.groups = [] #empty array at instantiation, filled and updated as merges are performed
		self.n = 0 #number of nodes in network
		self.associationMatrix = self.readFile(fileName) #simple graph 
		self.m = np.sum(self.associationMatrix)/2 #number edges
		self.regularization = (2*float(self.m)) #so we only have to calculate it once...
		self.eMatrix = self.get_e_matrix()
	def readFile(self,fileName):
		with open(fileName,'r') as f: #make 2d numpy array of appropriate size
			temp=-1
			lastNode=-1
			for line in f:
				line = line.split()
				temp = max(int(line[0]),int(line[1]))
				if (temp>lastNode):
					lastNode=temp #finds the highest numbered node 
			associationMatrix = np.zeros((lastNode,lastNode))
			self.n = int(lastNode) #assumes no gaps in node labelling
			f.seek(0,0)
			lines = f.readlines()
			for line in lines:
				line = line.split()
				associationMatrix[int(line[0])-1][int(line[1])-1] = 1 #make it undirected...
				associationMatrix[int(line[1])-1][int(line[0])-1] = 1
			for index in range(0,self.n): #self.groups is a list of lists
				self.groups.append([[]]) #add empty list for every vertex in graph
				self.groups[index][0] = index+1 #place every vertex in its own group
		f.close()
		return (associationMatrix)

	def inGroup(self,i): #returns group that node i belongs to
		group=0
		node = i
		for index in range (0,len(self.groups)):
			if (node in self.groups[index]):
				group=index
				break
		return group
		
	def get_e_matrix(self): #updates e matrix (used after merge is performed)
		numberGroups=0
		for index in range(0,len(self.groups)):
			if (self.groups[index][0]): #if group has member/members
				numberGroups+=1
		eMatrix = np.zeros((numberGroups,numberGroups))
		for r in range (0,numberGroups):
			for s in range (0,numberGroups):
				tempSum=0.0
				for i in range (0,self.n):
					if(self.inGroup(i+1) == r): 
						for j in range(0,self.n):
							if (self.associationMatrix[i][j]==1):
								if (self.inGroup(j+1) == s):
									tempSum+=1
				if (tempSum!=0):
					eMatrix[r][s]=(tempSum/self.regularization)
		return eMatrix
		
	def findDeltaQ(self,u,v): #returns delta Q between groups u and v
		a_u = np.sum(self.eMatrix[u])
		a_v = np.sum(self.eMatrix[v])
		return (2*(self.eMatrix[u][v]-(a_u*a_v)))

	def findGreatestDeltaQ(self): #returns (greatest delta Q,
		#				index of first group to merge, index of second group to merge)
		deltaQ = (float('-inf'),0,0)
		for index in range(0,len(self.groups)):
			for index2 in range (index+1,len(self.groups)):
				temp = self.findDeltaQ(index,index2)
				if (deltaQ[0] < temp):
					deltaQ = (temp,index,index2)
		return deltaQ

	def mergeGroups(self,r,s): #merge groups r and s into r, delete s
		r=int(r)
		s=int(s)
		for item in self.groups[s]:
			self.groups[r].append(item)
		del self.groups[s]
		return

	def getQ(self): #returns Q of network
		sum=0
		for index in range(0,len(self.groups)):
			sum2=0
			for index2 in range(0,len(self.groups)):
				sum2+=self.eMatrix[index][index2]
			sum+=self.eMatrix[index][index]-pow(sum2,2)
		return sum

def main():
	graph = Network(fileName)
	counter=0
	maxModularity=float('-inf')
	modularity=np.zeros((graph.n))
	while (len(graph.groups)>1):
		temp=graph.findGreatestDeltaQ() #temp is a triple, (max delta Q, group to merge, group to merge)
		Q = graph.getQ()
		graph.mergeGroups(temp[1],temp[2])
		modularity[counter] = Q
		print Q
		print graph.groups
		counter+=1
		graph.eMatrix=graph.get_e_matrix()
		if (maxModularity<graph.getQ()):
			maxModularity=graph.getQ()
			length = len(graph.groups)
			maxModularityGroups=[[]]*length
			#print graph.getQ()
			for index in range(0,length):
				maxModularityGroups[index] = graph.groups[index]
	print maxModularity
	print maxModularityGroups
	
	plt.ylabel('Modularity Q')
	plt.xlabel('number of merges')
	plt.xlim(1,34)
	plt.title('Modularity of Zachary Karate Club as a Function of Merges\n (using Greedy Agglomerative Algorithm)')
	plt.plot(modularity)
	plt.savefig("karateMerge.jpg")
	
main()