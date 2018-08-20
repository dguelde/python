# Donovan Guelde
# CSCI 5352 PS6 q1c

import numpy as np
from igraph import *
from sets import Set

def readFile():
	g = Graph(directed=True)
	
	with (open('cit-HepPh-dates.txt','r')) as f:
		next(f) #skip header row
		nodesByDate=[]
		for line in f:
			line=line.split()
			node,date=line[0],line[1]
			nodesByDate.append(node)
			
			
	

	edgeList=[]
	with (open('cit-HepPh.txt','r')) as f:
		
		nodes=Set()
		f.seek(0,0)
		next(f)
		next(f)
		next(f)
		next(f)
		for line in f:
			line=line.split()
			node,neighbor=line[0],line[1]
			if node not in nodes:
				nodes.add(node)
			if neighbor not in nodes:
				nodes.add(neighbor)

			edgeList.append((node,neighbor))
	for item in nodes:
		g.add_vertex(item)
	g.add_edges(edgeList)
	#how many vertices have date info?
	count=0.
	for item in nodes:
		if item in nodesByDate:
			count+=1
	return g,nodesByDate,nodes,count
	
	
			
def main():
	g,nodesByDate,nodes,haveDateInfo=readFile()
	interval=int(.1*haveDateInfo)
	print interval
	print nodesByDate[0]
	firstTenPercent=[]
	count=0
	counter=0
	firstTenTotal=0.
	while(count<interval):
		try:
			firstTenTotal+= Graph.degree(g,nodesByDate[counter],mode='in')
			node= nodesByDate[counter]
			degree= Graph.degree(g,nodesByDate[counter],mode='in')
			firstTenPercent.append((node,degree))
			count+=1
		except ValueError:
			pass
		counter+=1
	print "first 10% average:",firstTenTotal/interval

	counter=int(haveDateInfo)
	sum=0.
	count=0
	lastTenPercent=[]
	lastTenTotal=0.
	while(count<interval):
		try:
			lastTenTotal+= Graph.degree(g,nodesByDate[counter],mode='in')
			node= nodesByDate[counter]
			degree= Graph.degree(g,nodesByDate[counter],mode='in')
			lastTenPercent.append((node,degree))
			count+=1
		except ValueError:
			pass
		counter-=1
	print "last 10% average:",lastTenTotal/interval

	with(open('1cResults.txt','w')) as f:
		f.write('first ten percent:\n')
		f.write('\n'.join('%s %s' % x for x in firstTenPercent))

		f.write("total: "+str(firstTenTotal)+'\n')
		f.write("Average: "+str(firstTenTotal/interval))
		f.write('\n\n')
		f.write('last ten percent:\n')
		f.write('\n'.join('%s %s' % x for x in lastTenPercent))
		f.write('total: '+str(lastTenTotal)+'\n')
		f.write("Average: "+str(lastTenTotal/interval))


	



main()