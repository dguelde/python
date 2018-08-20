# Author: Donovan Guelde
# CSCI 5352 PS1
# script to generate plot from results of FB100
# references: numpy and pyplot documentation http://matplotlib.org/api/pyplot_api.html, http://docs.scipy.org/doc/numpy/reference/
#collaborators: None

import numpy as np 
import matplotlib.pyplot as pyplot

names = [] #an array to hold names of schools, where names[index] corresponds to plotResults[index]
with open('nameResults.txt','r') as f:
	for line in f:
		#name = [item.strip("'") for item in line.split()]
		names.append(line.strip('0123456789\r\n'))
f.close
print names

	


ku = []
kvkuRatio=[]
with open('plotResults.txt','r') as f:
	for line in f:
		kvTemp,kuTemp = line.split()
		ku.append(float(kuTemp))
		kvkuRatio.append(float(kvTemp)/float(kuTemp))
f.close()
		
for item in names:
	if (item == "Reed"):
		print "Banana"

		


"""
for item in ku,kvkuRatio:

	pyplot.scatter(ku,kvkuRatio)
	if (names[counter]=="Reed"):
		ax.annotate("Reed",xy=(ku[counter],kvkuRatio[counter]),xytext=(ku[counter-1],kvkuRatio[counter]+1))

	counter+=1
"""
pyplot.scatter(ku,kvkuRatio)
counter=0
for item in names:
	if (item == "Reed" or item == "Bucknell" or item == "Mississippi" or item == "Virginia" or item == "Berkeley"):
		print counter,"Reed"
		pyplot.annotate(item,xy=(ku[counter],kvkuRatio[counter]),xytext=(ku[counter],kvkuRatio[counter]),)
	counter+=1
pyplot.ylim(0)
pyplot.plot((30,120),(1,1))
pyplot.annotate("no paradox below this line, <ku> is larger than <kv>",xy = (40,1))
pyplot.ylabel("<kv>/<ku>")
pyplot.xlabel("<ku>")
pyplot.show()




