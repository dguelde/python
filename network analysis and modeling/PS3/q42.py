# Author: Donovan Guelde
# CSCI 5352 PS3 Question 4
# references: online documentation for numpy and igraph
# Collaborators: None

import numpy as np
import os
import igraph

class Network:
	def __init__(self,fileName):
		self.n = 0
		self.gender=[]
		self.status=[]
		self.major = []
		self.vertexDegree=[]
		self.associationMatrix = self.readFile(fileName)
		self.g = igraph.Graph.Adjacency((self.associationMatrix == 1).tolist())
		self.m = np.sum(self.associationMatrix)/2.0
		self.regularization = 1/(2*self.m) #caluclate once
	def readFile(self,fileName):
		with open("./facebook100txt/"+fileName+"_attr.txt",'r') as f: #get n and attributes from _attr.txt
			counter=0
			for line in f:
				counter+=1
			self.n = counter-1
			associationMatrix = np.zeros((self.n,self.n)) 
			self.gender = [0]*self.n #arrays to track gender, status, major and degree of vertexes
			self.status = [0]*self.n
			self.major = [0]*self.n
			self.vertexDegree = [0]*self.n 
			f.seek(0,0)
			f.next() #skip the label row
			counter=0
			for line in f: #populate the attribute arrays
				line = map(int,line.split())
				self.gender[counter] = int(line[2]) #gender of vertexes where index=vertex
				self.status[counter] = int(line[1]) #ditto...
				self.major[counter] = int(line[3])
				counter+=1
		f.close()

		with open("./facebook100txt/"+fileName+".txt",'r') as f: #construct association matrix
			lines = f.readlines()
			for line in lines:
				line = line.split()
				associationMatrix[int(line[0])-1][int(line[1])-1] = 1 
				associationMatrix[int(line[1])-1][int(line[0])-1] = 1 #make it undirected
		f.close()
		for index in range(0,self.n): #populate the vertex degree array, 
			self.vertexDegree[index] = np.sum(associationMatrix[index],axis=0)
		return associationMatrix

	def getQ(self,attribute): #returns Q of network
		if attribute == "gender":
			membership = self.gender
		if attribute == "status":
			membership = self.status
		if attribute == "major":
			membership = self.major
		Q = igraph.Graph.modularity(self.g,membership)
		return Q
	
	def calculateAssortativity(self):
		assortativityCoefficient=igraph.Graph.assortativity(self.g,self.vertexDegree)
		return assortativityCoefficient

def main():
	plotArray=np.empty((100,2)) #an array of points to plot
	nameArray=[""]*100 #array to hold names of schools where [index] corresponds to plotArray[index]
	nextUniversity = [2]
	lastUniversity = [2]
	genderModularity=[""]*100
	statusModularity=[""]*100
	majorModularity=[""]*100
	vertexAssortativity=[""]*100
	names=[""]*100
	nValues = [""]*100
	counter=0
	for file in os.listdir("./facebook100txt/"):
		if (file != ".DS_Store"):
			nextFile, fileExtension = os.path.splitext(file)
			nextUniversity = nextFile.split('_')
			if (str(nextUniversity[0]) != str(lastUniversity[0])):
				nextGraph = Network(nextUniversity[0])
				names[counter]=nextUniversity[0]
				nValues[counter]=nextGraph.n
				genderModularity[counter]=nextGraph.getQ("gender")
				statusModularity[counter]=nextGraph.getQ("status")
				majorModularity[counter]=nextGraph.getQ("major")
				vertexAssortativity[counter]=nextGraph.calculateAssortativity()
				nameArray[counter] = str(nextUniversity[0])
				lastUniversity=nextUniversity
				counter+=1
	with open("./results/genderModularity.txt","w") as f:
		np.savetxt(f,genderModularity,fmt='%s')
	f.close()
	with open("./results/statusModularity.txt","w") as f:
		np.savetxt(f,statusModularity,fmt='%s')
	f.close()
	with open("./results/majorModularity.txt","w") as f:
		np.savetxt(f,majorModularity,fmt='%s')
	f.close()
	with open("./results/vertexAssortativity.txt","w") as f:
		np.savetxt(f,vertexAssortativity,fmt='%s')
	f.close()
	with open("./results/names.txt","w") as f:
		np.savetxt(f,names,fmt='%s')
	f.close()
	with open("./results/nValues.txt","w") as f:
		np.savetxt(f,nValues,fmt='%s')
	f.close()

main()