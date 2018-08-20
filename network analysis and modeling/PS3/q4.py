# Author: Donovan Guelde
# CSCI 5352 PS3, some code reused from PS1
# references: online documentation for numpy
# Collaborators: None

import numpy as np
import os
import igraph



class Network:
	def __init__(self,fileName):
		self.n = 0
		self.gender=[]
		self.genderGroups = []
		self.status=[]
		self.statusGroups = []
		#self.major = []
		#self.majorGroups = []
		self.vertexDegree=[]
		self.associationMatrix = self.readFile(fileName)
		self.m = np.sum(self.associationMatrix)/2.0
		self.regularization = 1/(2*self.m)
		self.eMatrix = []
		#self.mnd = self.getMND()
	def readFile(self,fileName):
		with open("./facebook100txt/"+fileName+"_attr.txt",'r') as f: #uses attr file to initialize a 2d matrix of appropriate size
			
			counter=0
			for line in f:
				counter+=1
			associationMatrix = np.empty((counter,counter)) #in all arrays, index[0] is never used, left 0, to avoid 'off by one' errors, so vertex label == index

			self.gender = np.zeros((counter,1)) #array to track gender of vertexes
			#self.status = np.zeros((counter,1)) #array of status
			#self.major = np.zeros((counter,1)) #array of major
			self.vertexDegree = np.zeros((counter,1)) #array of vertex degree
			self.n = counter-1 #-1 for label row
			associationMatrix.fill(0)
			f.seek(0,0)
			f.next() #skip the label row
			tempGender=[]
			#tempStatus=[]
			#tempMajor=[]
			counter=1 #start at index 1 so vertex label == index
			for line in f: #populate the attribute arrays
				line = line.split()
				if(int(line[2]) not in tempGender):
					tempGender.append(int(line[2]))
				#if(int(line[1]) not in tempStatus):
				#	tempStatus.append(int(line[1]))
				#if(int(line[3]) not in tempMajor):
				#	tempMajor.append(int(line[3]))
				self.gender[counter] = int(line[2])
				#self.status[counter] = int(line[1])
				#self.major[counter] = int(line[3])
				counter+=1
			for item in tempGender: #make arrays of correct size for attributes
				self.genderGroups.append([])
			#for item in tempStatus:
			#	self.statusGroups.append([])
			#for item in tempMajor:
			#	self.majorGroups.append([])

		f.close()
		with open("./facebook100txt/"+fileName+".txt",'r') as f: #uses data file to build association matrix (assumes simple graph)
			lines = f.readlines()
			for line in lines:
				line = line.split()
				associationMatrix[int(line[0])][int(line[1])] = 1 #make it undirected...
				associationMatrix[int(line[1])][int(line[0])] = 1
		f.close()
		for index in range(1,counter): #populate the vertex degree array, skip [0] to match vertex labels
			self.vertexDegree[index] = np.sum(associationMatrix[index],axis=0)
		kAverageSum=0
		"""
		for index in range (0,self.n): 
			kAverageSum += self.vertexDegree[index]
		self.kAverage = kAverageSum/self.n
		"""
		return associationMatrix
	def inGroup(self,i,attribute): #returns group that node i belongs to
		group=0
		node = i
		if attribute=="gender":
			for index in range (0,len(self.genderGroups)):
				if (node in self.genderGroups[index]):
					group=index
					return group
		if attribute=="status":
			for index in range (0,len(self.statusGroups)):
				if (node in self.statusGroups[index]):
					group=index
					break
		if attribute=="major":
			for index in range (0,len(self.majorGroups)):
				if (node in self.majorGroups[index]):
					group=index
					break
		return group

	def get_e_matrix(self,attributeToUse): #updates e matrix (used after merge is performed)
		if attributeToUse=="gender":
			groups = self.genderGroups
			attribute = self.gender
		elif attributeToUse=="status":
			groups = self.statusGroups
			attribute = self.status
		elif attributeToUse=="major":
			groups = self.majorGroups
			attribute = self.major


		numberGroups=len(groups) #discover unique labels, and sort vertexes into labels
		groupLabels=[]
		for index in range(1,len(attribute)):
			if (attribute[index] not in groupLabels):
				groupLabels.append(attribute[index])
		for index in range(1,len(attribute)):
			for index2 in range(0,len(groupLabels)):
				if attribute[index] in groupLabels[index2]:
					groups[index2].append(index)

					

		eMatrix = np.zeros((numberGroups,numberGroups))
		for r in range (0,numberGroups):
			for s in range (0,numberGroups):
				tempSum=0.0
				for i in range (1,self.n):
					if(self.inGroup(i,attributeToUse) == r): 
						for j in range(1,self.n):
							if (self.associationMatrix[i][j]==1):
								if (self.inGroup(j,attributeToUse) == s):
									tempSum+=1.0
				if (tempSum!=0):
					eMatrix[r][s]=(tempSum*self.regularization)
		return eMatrix

	def getQ(self,attribute): #returns Q of network
		if attribute=="gender":
			groups = self.genderGroups
		if attribute == "status":
			groups = self.statusGroups
		if attribute == "major":
			groups = self.majorGroups
		self.eMatrix=self.get_e_matrix(attribute)
		sum1=0
		for index in range(0,len(groups)):
			sum2=np.sum(self.eMatrix[index])
			#for index2 in range(0,len(groups)):
			#	sum2+=self.eMatrix[index][index2]
			sum1+=self.eMatrix[index][index]-pow(sum2,2)
		return sum1

	#def getMND(self):
	#	associationMatrixSquared = np.linalg.matrix_power(self.associationMatrix,2)
	#	self.mnd = (np.sum(associationMatrixSquared)/np.sum(self.associationMatrix))
	#	return self.mnd
"""
	def calculateAssortativity(self):
		assortativityCoefficient=0.0
		sum1=0.0
		sum2=0.0
		attributeArray = self.vertexDegree
		for i in range(1,len(attributeArray)):#skip index 0
			iDegree=self.vertexDegree[i]
			for j in range(1,len(attributeArray)):
				if i==j: kroneckerDelta=1.0
				else: kroneckerDelta=0.0 
				KiKj = self.vertexDegree[i]*self.vertexDegree[j]*self.regularization
				sum1+=(self.associationMatrix[i][j] - KiKj)*(attributeArray[i]*attributeArray[j])
				sum2+=(iDegree*kroneckerDelta - KiKj)*(attributeArray[i]*attributeArray[j])
		assortativityCoefficient = sum1/sum2

		return assortativityCoefficient
"""
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
	counter=0
	for file in os.listdir("./facebook100txt/"):
		if (file != ".DS_Store"):
			nextFile, fileExtension = os.path.splitext(file)
			nextUniversity = nextFile.split('_')
			if (str(nextUniversity[0]) != str(lastUniversity[0])):
				if(nextUniversity[0]=="Caltech36"):
					nextGraph = Network(nextUniversity[0])
					#plotArray[counter][0] = float(nextGraph.mnd)
					#plotArray[counter][1] = float(nextGraph.kAverage)
					#print nextUniversity[0],nextGraph.getQ("gender")
					names[counter]=nextUniversity[0]
					genderModularity[counter]=nextGraph.getQ("gender")
					#statusModularity[counter]=nextGraph.getQ("status")
					#majorModularity[counter]=nextGraph.getQ("major")
					#vertexAssortativity[counter]=nextGraph.calculateAssortativity()[0]
					#print nextUniversity[0],nextGraph.calculateAssortativity()
					#nameArray[counter] = str(nextUniversity[0])
					#print nameArray
					lastUniversity=nextUniversity
					counter+=1
					print names
					print genderModularity
					#print statusModularity
					#print majorModularity
					#print vertexAssortativity
	#np.savetxt("plotResults.txt",plotArray)
	#with open("./results/genderModularity.txt","w") as f:
	#	np.savetxt(f,genderModularity,fmt='%s')
	#f.close()
	with open("./results/statusModularity.txt","w") as f:
		np.savetxt(f,statusModularity,fmt='%s')
	f.close()
	#with open("./results/majorModularity.txt","w") as f:
	#	np.savetxt(f,majorModularity,fmt='%s')
	#f.close()
	#with open("./results/vertexAssortativity.txt","w") as f:
	#	np.savetxt(f,vertexAssortativity,fmt='%s')
	#f.close()
	#with open("./results/names.txt","w") as f:
	#	np.savetxt(f,names,fmt='%s')
	#f.close()

main()