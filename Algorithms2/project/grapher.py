import numpy as np
import matplotlib.pyplot as plt
import os 

def readFiles(path,files):
	algTime = []
	bruteTime = []
	algResults = []
	bruteResults = []
	for files in os.listdir(path):
		with open(path+'/'+files, 'r') as f:
			if files[0:8]=="algTimes":
				for line in f:
					algTime.append(float(line))
			elif files[0:15]=="bruteForceTimes":
				for line in f:
					bruteTime.append(float(line))
			elif files[0:10] == "algResults":
				for line in f:
					algResults.append(float(line))
			elif files[0:17] == "bruteForceResults":
				for line in f:
					bruteResults.append(float(line))
    
	return (np.array(algTime),np.array(bruteTime),np.array(algResults),np.array(bruteResults))


def main():
	nValues=[3,4,5,6,7,8,9,10,11,12,13]
	averageAlgTime=np.zeros(11)
	averageBruteTime=np.zeros(11)
	maxAlgTime=np.zeros(11)
	maxBruteTime=np.zeros(11)
	minAlgTime=np.zeros(11)
	minBruteTime=np.zeros(11)
	averageAlgResult=np.zeros(11)
	averageBruteResult=np.zeros(11)
	maxAlgResult=np.zeros(11)
	maxBruteResult=np.zeros(11)
	minAlgResult=np.zeros(11)
	minBruteResult=np.zeros(11)
	d='.'
	directories = filter(lambda x: os.path.isdir(os.path.join(d, x)), os.listdir(d))
	for directory in directories:
		nTemp=directory[0:2]
		try:
			n=int(nTemp)
		except:
			n=int(nTemp[0])
		#print n

		#print directory
		files=("algResults{}.txt".format(n),"algTimes{}.txt".format(n),
				 "bruteForceResults{}.txt".format(n),"bruteForceTimes{}.txt".format(n))
		algTime,bruteTime,algResults,bruteResults=readFiles(directory,files)
		averageAlgTime[n-3]=np.mean(algTime)
		averageBruteTime[n-3]=np.mean(bruteTime)
		averageBruteResult[n-3]=np.mean(bruteResults)
		averageAlgResult[n-3]=np.mean(algResults)
		maxBruteResult[n-3]=np.amax(bruteResults)
		minBruteResult[n-3]=np.amin(bruteResults)
		maxAlgResult[n-3]=np.amax(algResults)
		minAlgResult[n-3]=np.amin(algResults)
		maxBruteTime[n-3]=np.amax(bruteTime)
		maxAlgTime[n-3]=np.amax(algTime)
		minBruteTime[n-3]=np.amin(bruteTime)
		minAlgTime[n-3]=np.amin(algTime)
	print "algResults"
	ratio=[]
	expectedRatio=[]
	for index in range(0,11):
		#print "averageAlgResult",averageAlgResult[index],"averageBruteResult",averageBruteResult[index],"ratio",averageAlgResult[index]/averageBruteResult[index]
		print averageBruteTime[index]," & ", averageAlgTime[index]," & ", averageBruteTime[index]/averageAlgTime[index]," & ", float(np.math.factorial(index+3))/float(((index+3)**3))," & ",(averageBruteTime[index]/averageAlgTime[index])/(float(np.math.factorial(index+3))/float(((index+3)**3))),"\\\\"
		#print np.math.factorial(index+3.)
		#print ((index+3.)**3)
		#print float(np.math.factorial(index+3))/float(((index+3)**3))
		ratio.append(averageAlgResult[index]/averageBruteResult[index])
		expectedRatio.append(np.math.factorial(index+3)/((index+3)**3))
		print "\\hline"

	my_xticks = [3,4,5,6,7,8,9,10,11,12,13]
	plt.semilogy(my_xticks,ratio)
	plt.semilogy(my_xticks,expectedRatio)
	plt.title('Ratio of (Brute Force Runtimeime)/(Algorithm Runtime) as a function of n')
	plt.xlabel('Number of Vertexes in Graph G=(V,E)')
	plt.ylabel('OPT/ALG')
	
	#plt.show()
	
	"""
	my_xticks = [3,4,5,6,7,8,9,10,11,12,13]
	plt.plot(my_xticks,maxAlgResult,label='Maximum Algorithm Cost')
	plt.plot(my_xticks,maxBruteResult,label='Maximum Brute Force Cost')
	plt.plot(my_xticks,averageAlgResult,label='Average Algorithm Cost')
	plt.plot(my_xticks,averageBruteResult,label='Average Brute Force Cost')
	plt.plot(my_xticks,minAlgResult,label='Minimum Algorithm Cost')
	plt.plot(my_xticks,minBruteResult,label='Minimum Brute Force Cost')
	plt.title('Results of Christophides Algorithm\nand Brute Force Solution of TSP')
	plt.xlabel('Number of Vertexes in Graph G=(V,E)')
	plt.ylabel('Path Cost')
	plt.legend(loc='lower right')
	plt.show()
	plt.plot(my_xticks,maxBruteTime,label="Maximum Brute Force Time")
	plt.semilogy(my_xticks,averageBruteTime,label="Average Brute Force Time")
	plt.plot(my_xticks,minBruteTime,label="Minimum Brute Force Time")
	plt.plot(my_xticks,maxAlgTime,label="Maximum Algorithm Time")
	plt.semilogy(my_xticks,averageAlgTime,label="Average Algorithm Time")
	
	plt.plot(my_xticks,minAlgTime,label="Minimum Algorithm Time")
	
	
	plt.title('Runtime of Christophides Algorithm\nand Runtime of Brute Force Solution to TSP')
	plt.xlabel('Number of Vertexes in Graph G=(V,E)')
	plt.ylabel('Observed Runtime in Seconds\n(logarithmic scale)')
	plt.legend(loc='upper left')
	plt.show()
	"""


main()