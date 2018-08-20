# Author: Donovan Guelde
# CSCI-3202 Assignment 3
# Implement A* search algorithm with 2 different heuristics

import sys
from math import sqrt


class Node(object):
		def __init__(self, locationX,locationY,value):
			self.xCoord = locationX
			self.yCoord = locationY
			self.location = (locationX,locationY) # a tuple from x,y values 
			self.value = value # 0=flat,1=mountain,2=wall
			self.g = 0 #distance back to start
			self.h = 0 #estimated distance to goal (heuristic)
			self.f = 0 # f+g
			parent = None # to remember path back to start

class Graph(object):
	def __init__(self):
		self.Null = Node(-1,-1,-1) # a 'Null' node
		self.open = [] #list of 'opened' nodes
		self.closed = [] #list of 'closed' nodes
		self.nodes = [] #and keep a 'master list' of nodes
		self.start = self.Null
		self.goal = self.Null
		self.height = 0
		self.width = 0
	def getNode(self,x,y): #find a node from its coordinates
		if (x<1 or x > self.width or y<1 or y>self.height): #if x,y are out of bounds, return null node
			return self.Null
		length = len(self.nodes)
		for index in range (0,length): #search node list for a node with desired coordinates
			if (self.nodes[index].location == (x,y)):
				return self.nodes[index]
	def getAdjacent(self,x,y):
		unfilteredAdjacentNodes = []
		adjacentNodes = []
		#gets all nodes surrounding (x,y), including Empty 
		unfilteredAdjacentNodes.append(self.getNode(x-1,y-1))
		unfilteredAdjacentNodes.append(self.getNode(x,y-1))
		unfilteredAdjacentNodes.append(self.getNode(x+1,y-1))
		unfilteredAdjacentNodes.append(self.getNode(x-1,y))
		unfilteredAdjacentNodes.append(self.getNode(x+1,y))
		unfilteredAdjacentNodes.append(self.getNode(x-1,y+1))
		unfilteredAdjacentNodes.append(self.getNode(x,y+1))
		unfilteredAdjacentNodes.append(self.getNode(x+1,y+1))
		#filter out empty nodes
		length = len(unfilteredAdjacentNodes)
		for index in range (0,length):
			if (unfilteredAdjacentNodes[index].value >=0 and unfilteredAdjacentNodes[index].value < 3):
				adjacentNodes.append(unfilteredAdjacentNodes[index])
		return adjacentNodes

def readInFile(matrix):
	#numberCLAs = len(sys.argv)
	matrixFile = sys.argv[1]
	lineIndex = 0
	file = open (matrixFile,'r')
	for line in file:
		itemsAdded = False
		matrix.append([])
		for item in line:
			#line.replace(" ","")
			if (item!=" " and item!="\n"): #don't add a node with value " " or "\n"
				matrix[lineIndex].append(int(item))
				itemsAdded = True
			if (itemsAdded == False):
				matrix.pop() #if no items added during last iteration, remove that list (don't add empty lines)
		lineIndex+=1

def buildWorld(graph,matrix): #takes info from matrix and creates the nodes and graph
	height = graph.height = len(matrix) #height of matrix = number of items in first (1D) list
	width = graph.width = len(matrix[0]) # width = number of items in any given (2D) list, assuming matrix is rectangular
	for x in range (0,height):
		for y in range (0,width):
			newNode = Node(y+1,8-x,matrix[x][y]) # transform to standard x,y coordinates (1,1) in lower left
			graph.nodes.append(newNode) #add to the graph's node list
			if (newNode.location == (1,1)): #1,1 is start
				graph.start = newNode
			if (newNode.location == (width,height)): # node at (height,width) is the goal
				graph.goal = newNode
	graph.open.append(graph.start) # add starting location to open list
	
def implementHeuristic(heuristic, graph):
	if heuristic == "1":
		firstHeuristic(graph)
	if heuristic == "2":
		nextHeuristic(graph)
	else:
		print "Heuristic choice not valid:"
		print "usage: python.<filename.py> <world file.txt> <heuristic choice (\"1\" or \"2\")"

	
def firstHeuristic(graph): #calculates Manhattan distance to goal
	graph.start.g = 0 #start is 0 units from start
	index = 0
	while (index<len(graph.nodes)):
		graph.nodes[index].h = ((graph.width - graph.nodes[index].xCoord) + (graph.height - graph.nodes[index].yCoord))*10
		index +=1
	graph.start.f = graph.start.h #populate start node f distance = h

def nextHeuristic(graph): #calculates straight-line cost using average cost over entire board
	numberSquares = graph.height * graph.width
	sumCost = 0
	length = len(graph.nodes)
	for index in range(0,length):
		sumCost += (10 + graph.nodes[index].value*10)
	averageCost = sumCost/numberSquares
	index = 0
	while (index<len(graph.nodes)):
		node = graph.nodes[index]
		node.h = int(sqrt((graph.width - node.xCoord)**2 + (graph.height - node.yCoord)**2))*averageCost
		index +=1
	graph.start.f = graph.start.h #populate start node f distance = h


def pathFinder(graph): #find optimal path using A* algorithm
	
	solutionFound = False 
	while (solutionFound == False): #repeat until goal node closed
		length = len(graph.open)
		minF = sys.maxint 
		for index in range (0,length):# get next node from open list with lowest F value
			if (graph.open[index].f < minF):
				nextNode = graph.open[index]
				minF = nextNode.f
		if (nextNode == graph.goal): #solution found
			solutionFound = True
		graph.open.remove(nextNode) #remove this node from open list
		graph.closed.append(nextNode) # put in closed list
		adjacentList = graph.getAdjacent(nextNode.xCoord,nextNode.yCoord) # get list of adjacent nodesnodes
		length = len(adjacentList) 
		for index in range (0,length): #check status of adjacent nodes (already open, mountain, not valid)
			stepDistance = 0 #determine movement cost to adjacent node
			if (adjacentList[index].value == 1): #penalty for mountains
				stepDistance = 10
			if ((adjacentList[index].xCoord == nextNode.xCoord) or (adjacentList[index].yCoord == nextNode.yCoord)):
				stepDistance +=10 #distance to adjacent node = 10 if not diagonal
			else:
				stepDistance +=14 #distance = 14 if diagonal
			if (adjacentList[index].value == 2 or adjacentList[index] in graph.closed): 
				continue #if adjacent node is a wall or closed, ignore it
			
			if (adjacentList[index] in graph.open): #if node is already in open list, check if this is shorter route
				if (adjacentList[index].g < nextNode.g + stepDistance):
					continue #this path is longer than what is already calculated for this node, so ignore
				else: #found a shorter path, update the node's info, and re-add
					graph.open.remove(adjacentList[index])
					adjacentList[index].parent = nextNode
					adjacentList[index].g = nextNode.g + stepDistance
					adjacentList[index].f = adjacentList[index].g + adjacentList[index].h
					graph.open.append(adjacentList[index])
			else: #if the adjacent node is not already in open list
				adjacentList[index].parent = nextNode # update parent
				adjacentList[index].g = nextNode.g + stepDistance #update g distance
				graph.open.append(adjacentList[index]) #add to open list

def generatePath(graph): #using Node.parent info, create a list representing optimal path
	path = []
	node = graph.goal
	while (node != graph.start):
		node = node.parent
		path.append(node)
	return path
		
def main():
	if (len(sys.argv) < 3):
		print
		print "usage: python <filename.py> <heuristic filename.txt> <\"1\" or \"2\">"
		return
	worldMatrix = [] #initialize empty graph
	readInFile(worldMatrix) # read in matrix from file
	graph = Graph() #initialize empty graph
	buildWorld(graph,worldMatrix) #create nodes, populate graph with info from input file
	heuristic = sys.argv[2]
	implementHeuristic(heuristic,graph)
	pathFinder(graph) #run pathfinding
	path = generatePath(graph) #get optimal path
	print
	print "Pathfinding Results for",sys.argv[1],"heuristic",sys.argv[2]
	print "path:",
	length = len(path)
	for index in range (length-1,-1,-1): #path list runs from goal to start, so print from last item to first
		print path[index].location,"->",
	print graph.goal.location
	print len(graph.open),"nodes open"
	print len(graph.closed),"nodes closed"
	print
	


	

main()

