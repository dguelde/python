# examine samples
import csv
from collections import defaultdict
import operator


spoiler=[]
page=[]
trope=[]
counter=0
with (open("../data/spoilers/train.csv", 'rb')) as f:
	reader=csv.reader(f,delimiter=',')
	next(reader,None)

	for row in reader:
		spoiler.append(row[1])
		page.append(row[3])
		trope.append(row[4])
			
f.close()
winner=[]
dictionary={}
dataList=[]
seenThis=[]
for item in page:
	if item not in seenThis:
		seenThis.append(item)
	data=(item,0,0)
	counter=0
	trueCount=0
	falseCount=0
	change=0
	for item2 in spoiler:
		if (item2=='False' and page[counter]==item):
			falseCount+=1
			change=1
		if (item2=='True' and page[counter]==item):
			trueCount+=1
			change=1
		counter+=1
	data=(trueCount,falseCount,abs(trueCount-falseCount))
	dictionary[item]=data
counter=0
for item in dictionary:
	true=dictionary[item][0]
	false=dictionary[item][1]
	diff=dictionary[item][2]
	if(diff>((true+false)*.5)):
		#print dictionary[item][2],item,'true',dictionary[item][0],'false',dictionary[item][1]
		counter+=1
		if(item not in winner):
			winner.append(item)
print counter,len(seenThis)

dictionary={}
dataList=[]
seenThis=[]
for item in trope:
	if item not in seenThis:
		seenThis.append(item)
	data=(item,0,0)
	counter=0
	trueCount=0
	falseCount=0
	change=0
	for item2 in spoiler:
		if (item2=='False' and trope[counter]==item):
			falseCount+=1
			change=1
		if (item2=='True' and trope[counter]==item):
			trueCount+=1
			change=1
		counter+=1
	data=(trueCount,falseCount,abs(trueCount-falseCount))
	dictionary[item]=data
counter=0
for item in dictionary:
	print dictionary[item][2],item,'true',dictionary[item][0],'false',dictionary[item][1]
	true=dictionary[item][0]
	false=dictionary[item][1]
	diff=dictionary[item][2]
	if(diff>((true+false)*.5)):
		if(item not in winner):
			winner.append(item)
		
		counter+=1
print len(winner)
print len(seenThis)



