# Author: Donovan Guelde
# CSCI 5352 PS1
# references: online documentation for numpy
# Collaborators: None

import numpy as np
import os



class Network:
	def __init__(self,fileName):
		self.n = 0
		self.kAverage = 0
		self.degreeVector=[]
		self.associationMatrix = self.readFile(fileName)
		self.m = np.sum(self.associationMatrix)/2
		self.mnd = self.getMND()
	def readFile(self,fileName):
		with open("./CSCI5352_Data/facebook100txt/"+fileName+"_attr.txt",'r') as f: #uses attr file to initialize a 2d matrix of appropriate size
			counter=0
			for line in f:
				counter+=1
			associationMatrix = np.empty((counter,counter))
			self.n = counter-1 #-1 for label row
			associationMatrix.fill(0)
		f.close()
		with open("./CSCI5352_Data/facebook100txt/"+fileName+".txt",'r') as f: #uses data file to build association matrix (assumes simple graph)
			lines = f.readlines()
			for line in lines:
				line = line.split()
				associationMatrix[int(line[0])][int(line[1])] = 1 #make it undirected...
				associationMatrix[int(line[1])][int(line[0])] = 1
		f.close()
		self.degreeVector = np.sum(associationMatrix,axis=0)
		kAverageSum=0
		for index in range (1,self.n+1): 
			
			kAverageSum += self.degreeVector[index]
		self.kAverage = kAverageSum/self.n
			
		return associationMatrix
	def getMND(self):
		associationMatrixSquared = np.linalg.matrix_power(self.associationMatrix,2)
		self.mnd = (np.sum(associationMatrixSquared)/np.sum(self.associationMatrix))
		return self.mnd

def main():
	plotArray=np.empty((100,2)) #an array of points to plot
	nameArray=[""]*100 #array to hold names of schools where [index] corresponds to plotArray[index]
	nextUniversity = [2]
	lastUniversity = [2]
	counter=0
	for file in os.listdir("./CSCI5352_Data/facebook100txt/"):
		if (file != ".DS_Store"):
			
			nextFile, fileExtension = os.path.splitext(file)
			nextUniversity = nextFile.split('_')
			if (str(nextUniversity[0]) != str(lastUniversity[0])):
				nextGraph = Network(nextUniversity[0])
				plotArray[counter][0] = float(nextGraph.mnd)
				plotArray[counter][1] = float(nextGraph.kAverage)
				
				nameArray[counter] = str(nextUniversity[0])
				print nameArray
				lastUniversity=nextUniversity
				counter+=1
	np.savetxt("plotResults.txt",plotArray)
	with open("nameResults.txt","w") as f:
		np.savetxt(f,nameArray,fmt='%s')

main()