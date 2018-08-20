import numpy as np
import math
import matplotlib.pyplot as plt
import os
from itertools import islice

def plotResults(epidemicLength,epidemicSize,pValues): #make some graphs...
	N=1000
	C=8
	E=0.0
	plt.plot(pValues,epidemicSize)
	plt.title('Average Epidemic Size\n(n={}, C={}, Epsilon={})'.format(N,int(C),E))
	plt.xlabel('P')
	plt.ylabel('Epidemic Size')
	#plt.show()
	plt.savefig('size.jpg'.format(E))
	plt.close()


	tMax=0
	maxIndex=0
	counter=0.
	for item in epidemicLength:
		if item>tMax:
			tMax=item
			maxIndex=counter
		counter+=1
	maxIndex = float(maxIndex/200)
	plt.plot(pValues,epidemicLength)
	plt.annotate('Maximum Length {} at p={}'.format(tMax,maxIndex),xy=(maxIndex,tMax))
	plt.title('Average Epidemic Length\n(n={}, C={}, Epsilon={})'.format(N,int(C),E))
	plt.xlabel('P')
	plt.ylabel('Epidemic Length')
	
	plt.axhline(y=math.log(N),xmin=0,xmax=1,color='r',ls='dashed')
	#plt.show()
	plt.savefig('length.jpg'.format(E))
	plt.close()

def readDataFiles():
	
	lengthData=[]
	sizeData = []
	
	
	
	filename = 'FinalE0.0length.txt'
	with (open(filename,'r')) as f:
		lines=f.readlines()
		for line in lines:
			lengthData.append(float(line))
	f.close()

	filename = 'FinalE0.0size.txt'
	with (open(filename,'r')) as g:
		lines=g.readlines()
		for line in lines:
			sizeData.append(float(line))
	g.close()

		

	return lengthData,sizeData



def main():
	
	p=[]
	counter=0
	for index in range(0,200):
		counter=float(format(counter,'.3f'))
		p.append(counter)
		counter+=0.005
	
	
	lengthData,sizeData = readDataFiles()
	#print lengthData
	plotResults(lengthData,sizeData,p)

	
	"""
	"eValues,lengthData,sizeData = readDataFiles(p)
	"print 'length',lengthData,'\n'
	"print "size",sizeData
	"""


main()