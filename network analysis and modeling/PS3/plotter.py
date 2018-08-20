# Donovan Guelde
# CSCI 5352 PS3
# Plotter for PS3, Q4

import numpy as np
import matplotlib.pyplot as plt

def readData(fileName):
	counter=0
	with open(fileName,'r') as f:
		for line in f:
			counter+=1
		f.seek(0,0)
		data = np.zeros((counter)) # array to  hold data
		counter=0
		maxDataPoint = float('-inf')
		maxIndex=0
		minDataPoint = float('inf')
		nimIndex=0
		for line in f:
			data[counter] = float(line)
			if data[counter] > maxDataPoint:
				maxDataPoint=data[counter]
				maxIndex=counter
			if data[counter] < minDataPoint:
				minDataPoint=data[counter]
				minIndex=counter
			counter+=1
	return data, int(maxIndex), int(minIndex)

def readNames(fileName):
	counter=0
	with open(fileName,'r') as f:
		for line in f:
			counter+=1
		f.seek(0,0)
		data = [""]*counter
		counter=0
		for line in f:
			data[counter] = line
			counter+=1
	return data
def getData(attribute):
	if attribute == "major":
		data, maxPoint, minPoint = readData("./results/majorModularity.txt")
	if attribute == "status":
		data, maxPoint, minPoint = readData("./results/statusModularity.txt")
	if attribute == "vertex":
		data, maxPoint, minPoint = readData("./results/vertexAssortativity.txt")
	return data,int(maxPoint),int(minPoint)



def main():
	
	nArray,nMax,nMin = readData("./results/nValues.txt")
	names = readNames("./results/names.txt")
	attributes = ["major","status","vertex"]
	
	for item in attributes: 
		# scatter plots
		data,maxPoint,minPoint = getData(item)
		if item == "vertex":
			plt.ylabel('Vertex Degree Assortativity')
		else:
			plt.ylabel(item+" modularity")
		plt.xlabel('Network Size, n')
		plt.title('Network Size vs '+item)
		plt.scatter(nArray,data)
		if item=='major':
			plt.ylim(-.01,.14)
		plt.text(float(nArray[maxPoint]),float(data[maxPoint]),names[maxPoint])
		plt.plot(float(nArray[maxPoint]),float(data[maxPoint]),'o',mfc='red')
		plt.text(float(nArray[minPoint]),float(data[minPoint]),names[minPoint])
		plt.plot(float(nArray[minPoint]),float(data[minPoint]),'o',mfc='red')
		plt.text(float(nArray[nMax]),float(data[nMax]),names[nMax])
		plt.plot(float(nArray[nMax]),float(data[nMax]),'o',mfc='red')
		plt.text(float(nArray[nMin]),float(data[nMin]),names[nMin])
		plt.plot(float(nArray[nMin]),float(data[nMin]),'o',mfc='red')
		plt.axhline(0,linestyle='dashed')
		plt.xscale('log')
		plt.savefig(item+".jpg")
		plt.clf()
		plt.close()

		# histograms
		plt.hist(data)
		if item=='status':
			plt.xlim(-.05,.30)
		if item=='major':
			plt.xlim(-.01,.14)
		plt.title('Histogram of '+item+' Assortativity')
		plt.xlabel('Assortativity ('+item+')')
		plt.ylabel('Frequency')
		plt.axvline(0,linestyle='dashed', color='red')
		plt.savefig(item+"Hist.jpg")
		plt.clf()
		plt.close()

main()