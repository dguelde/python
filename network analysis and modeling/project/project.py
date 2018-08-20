# Donovan Guelde
# CSCI 5352, project
# recreate network from original paper by Marcel Salathe, Maria Kazandjieva, Jung Woo Lee, Philip Levis, Marcus W. Feldman, and James H. Jones

import numba
from numba import autojit
import numpy as np
import random
import time


PATH="./data/"
POPULATIONSIZE=788



def getAssociationMatrix():
	nodes=[]
	with open(PATH+"sd02.txt","r") as f: #node 548 doesn't exist.....
		
		associationMatrix=np.zeros((POPULATIONSIZE,POPULATIONSIZE))
		for line in f:
			line=line.split()
			node=int(line[0])
			neighbor=int(line[1])
			if(node>548): #548 doesn't exist...
				node-=1
			if(neighbor>548):
				neighbor-=1
			node-=1 #index from 0
			neighbor-=1
			weight=int(line[2])
			associationMatrix[node][neighbor]+=weight
			associationMatrix[neighbor][node]+=weight
	f.close()
	return associationMatrix


#@numba.jit
def epidemic(associationMatrix):
	results=np.zeros(POPULATIONSIZE)
	zeros=np.zeros(POPULATIONSIZE)
	maximums=np.zeros(POPULATIONSIZE)
	minimums=np.zeros(POPULATIONSIZE)
	for patientZero in xrange(POPULATIONSIZE):
		average=np.zeros(1000)
		zeroCount=0
		
		maximum=0
		minimum=1000
		for repetition in xrange(1000):
			susceptible=np.ones(POPULATIONSIZE) #all susceptible at start
			exposed=np.zeros(POPULATIONSIZE) #none exposed
			infectious=np.zeros(POPULATIONSIZE) #none infectious
			recovered=np.zeros(POPULATIONSIZE) #none recovered
			exposed[patientZero]=random.randint(1,9) #random incubation time
			susceptible[patientZero]=0 #patient zero not susceptible any more
			numberOfSpreaders=1 #numberOfSpreaders = infected + exposed (anyone who can potentially spread disease)
			while numberOfSpreaders!=0: 
				exposedPersons=np.where(exposed>0)[0]
				for exposee in exposedPersons:
					exposed[exposee]-=1 #count down incubation time
					if exposed[exposee]==0: # now contageous
						infectious[exposee]=1

				spreaders=np.where(infectious>0)[0] 
				for spreader in spreaders:
					for spreader in spreaders: #check for recovery
						if infectious[spreader]==12: #from paper, all recover by this point
							infectious[spreader]=0
							recovered[spreader]=1
						else:
							recoveryProbability=(1-.95**infectious[spreader]) #from paper

							if recoveryProbability>random.random(): #recovered
								infectious[spreader]=0
								recovered[spreader]=1
						if infectious[spreader]>=1: #if not recovered, increment by 1 (like umbers of days sick)
							infectious[spreader]+=1
				spreaders=np.where(infectious>0)[0] #update contagious list after recoveries
				for spreader in spreaders:
					neighbors=list(np.where(associationMatrix[spreader]>0)[0])
					susceptibles=list(np.where(susceptible==1)[0])
					potentialNewInfections=np.intersect1d(neighbors,susceptibles) #only neighbors who are susceptible
					for person in potentialNewInfections:
						#print person
						infectionProbability=(1-(1-0.003)**associationMatrix[spreader][person])
						#infectionProbability=.05
						immunity=random.random()
						if  immunity < infectionProbability: #didn't wash hands....
							exposed[person]=random.randint(2,9) #incubate 1-4 days
							susceptible[person]=0
							
						else: susceptible[person]=0 #???

				numberOfSpreaders=np.sum(infectious)+np.sum(exposed)
			average[repetition]=np.sum(recovered)
			if(average[repetition]==1):
				zeroCount+=1
			if(average[repetition]>maximum): maximum=average[repetition]
			if(average[repetition]<minimum): minimum=average[repetition]
		print "patient Zero:",patientZero,"average epidemic size",np.average(average),"zero count:",zeroCount,"maximum",maximum,"minimum",minimum
		results[patientZero]=np.average(average)
		zeros[patientZero]=zeroCount
		maximums[patientZero]=maximum
		minimums[patientZero]=minimum
	np.savetxt("results.txt",results)
	np.savetxt("zeros.txt",zeros)
	np.savetxt("maximums.txt",maximums)
	np.savetxt("minimums.txt",minimums)
	return



if __name__ == "__main__":
	
	associationMatrix=getAssociationMatrix()
	count=0
	print "numba"
	temp=0
	epidemic(associationMatrix)
	
	






