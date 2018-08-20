# Author: Donovan Guelde
# CSCI-3202 Fall 2015
# Assignment 5
# Implement an MDP algorithm with Bellman's equation

# I tested this code with epsilon = 10,1,.5,.1, and did not detect any change in path.  All epsilon values
# resulted in the same optimal path and same utility of nodes along that path.  Seems fishy...but
# I couldn't find anything wrong...

import csv
import sys

class Node(object):
        def __init__(self, locationX,locationY,value,reward):
            self.xCoord = locationX
            self.yCoord = locationY
            self.location = (locationX,locationY) # a tuple from x,y values 
            self.value = value #value from text file, 0 = flat, 1 = mountain, 2 = wall, 3 = snake, 4 = barn
            self.reward = reward # reward of occupying this location; flat = 0, mountain = -1, snake = -2, finish = 50
            self.utility = 0 # utility of this location, initially set to 0
          
            self.parent = None
            self.UValue  = 0 #  for comparing change in utility between iterations
            self.UPrimeValue = 0 # 

class Graph(object):
    def __init__(self):
        self.Null = Node(-1,-1,-1,-1) # a 'Null' node
        self.nodes = [] #and keep a 'master list' of nodes
        self.start = self.Null
        self.goal = self.Null
        self.height = 0
        self.width = 0

    def getNode(self,x,y): #find and return a node from its coordinates
        if (x<1 or x > self.width or y<1 or y>self.height): #if x,y are out of bounds, return null node
            return self.Null
        length = len(self.nodes)
        for index in range (0,length): #search node list for a node with desired coordinates
            if (self.nodes[index].location == (x,y)):
                return self.nodes[index]

    def getAdjacent(self,x,y): #gets all nodes surrounding (x,y) including walls, excluding non-valid nodes
        unfilteredAdjacentNodes = []
        adjacentNodes = []
        
        if (y>1):
            unfilteredAdjacentNodes.append(self.getNode(x,y-1))
        if (x>1):
            unfilteredAdjacentNodes.append(self.getNode(x-1,y))
        if (x<self.width):
            unfilteredAdjacentNodes.append(self.getNode(x+1,y))
        if (y<self.height):
            unfilteredAdjacentNodes.append(self.getNode(x,y+1))
        #filter out invalid nodes/walls
        length = len(unfilteredAdjacentNodes)
        for index in range (0,length):
            if (unfilteredAdjacentNodes[index].value >=0 and unfilteredAdjacentNodes[index].value != 2):
                adjacentNodes.append(unfilteredAdjacentNodes[index]) #not a wall, not invalid, so put it in the 'valid' list
        return adjacentNodes

    def isValidNeighbor(self,source, destination): #check to see if the node in question can be occupied
        validLocations = self.getAdjacent(source.xCoord,source.yCoord)
        if (destination in validLocations): return True #true if the horse can move to destination
        else: return False #false if cannot move there (wall or off the map)

    
    def getNorthNeighbor(self,sourceNode): #finds and returns the node to the north
        return self.getNode(sourceNode.xCoord,sourceNode.yCoord+1)

    def getEastNeighbor(self,sourceNode):
        return self.getNode(sourceNode.xCoord+1,sourceNode.yCoord)

    def getSouthNeighbor(self,sourceNode):
        return self.getNode(sourceNode.xCoord,sourceNode.yCoord-1)

    def getWestNeighbor(self,sourceNode):
        return self.getNode(sourceNode.xCoord-1,sourceNode.yCoord)


    def getExpectedUtilities(self,sourceNode): #returns an array, index[0] = EU for moving north, [1] = EU east, [2] = EU south, [3] = EU west

        #first gets surrounding nodes with easy-to-remember labels
        northNeighbor = self.getNorthNeighbor(sourceNode)
        eastNeighbor = self.getEastNeighbor(sourceNode)
        southNeighbor = self.getSouthNeighbor(sourceNode)
        westNeighbor = self.getWestNeighbor(sourceNode)
        
        #if the neighboring nodes can be occupied, set the utility for moving in that direction (u' + reward).  Oherwise, reward = reward of current location
        if (self.isValidNeighbor(sourceNode,northNeighbor)):
            northUtility = northNeighbor.utility #utilty of moving north = utility of north neighbor + reward of north node
        else: northUtility = sourceNode.reward #if north neighbor cannot be occupied, reward = reward of my current location
        if (self.isValidNeighbor(sourceNode,eastNeighbor)): #same for east, south, and west neighbors...
            eastUtility = eastNeighbor.utility 
        else: eastUtility = sourceNode.reward
        if (self.isValidNeighbor(sourceNode,southNeighbor)):
            southUtility = southNeighbor.utility 
        else: southUtility = sourceNode.reward
        if (self.isValidNeighbor(sourceNode,westNeighbor)):
            westUtility = westNeighbor.utility 
        else: westUtility = sourceNode.reward
        #calculate expected utility for attempting to move in any direction
        northEU = .8*northUtility + .1*westUtility + .1*eastUtility 
        eastEU = .8*eastUtility+ .1*northUtility + .1*southUtility
        southEU = .8*southUtility + .1*westUtility + .1*eastUtility
        westEU = .8*westUtility+ .1*northUtility + .1*southUtility
        EUArray = [northEU,eastEU,southEU,westEU] #array of expected utilities [north,east,south,west]
        return EUArray

    def updateExpectedUtilities(self,gamma): #updates EU of all nodes, starting from goal(upper right) ending at orogin lower left
        self.goal.utility = 50
        self.goal.reward = 50 #goal value
        length = len(self.nodes)
        for index in range(0,length):
            tempNode = self.nodes[index] #iterate through nodes in no particular order, not very efficient, I think, but it works
            if (tempNode.value != 50 and tempNode.value != 2): #don't recalculate goal or walls
                utilityMatrix = self.getExpectedUtilities(tempNode) #gets an array of expected utilities for surrounding nodes
                tempNode.utility = gamma*max(utilityMatrix) + tempNode.reward #utility = max utility of neighbors + reward
                tempNode.UPrimeValue = tempNode.utility #put this value in the 'UPrime' field for later comparison
                arrayIndex = max(utilityMatrix) #figure out which direction is best choice...already found the max value, now need to get the direction
                if (arrayIndex == utilityMatrix[0]): tempNode.parent = self.getNorthNeighbor(tempNode) #best reward is north, so parent = north neighbor
                elif (arrayIndex == utilityMatrix[1]): tempNode.parent = self.getEastNeighbor(tempNode) #best reward is east
                elif (arrayIndex == utilityMatrix[2]): tempNode.parent = self.getSouthNeighbor(tempNode) #best reward is south
                elif (arrayIndex == utilityMatrix[3]): tempNode.parent = self.getWestNeighbor(tempNode) #best reward is west
                

def readInFile(matrix): #reads text file and copies info into matrix using CSV 
    matrixFile = sys.argv[1]
    lineIndex = 0
    file = csv.reader(open (matrixFile),delimiter=" ")
    for line in file:
        itemsAdded = False
        matrix.append([])
        for item in line:
            if (item!=" " and item!="\n"): #don't add a node with value " " or "\n"
                matrix[lineIndex].append(int(item))
                itemsAdded = True
            if (itemsAdded == False):
                matrix.pop() #if no items added during last iteration, remove that list (don't add empty lines)
        lineIndex+=1

def buildWorld(graph,matrix): #takes info from matrix and creates the nodes and graph
    height = graph.height = len(matrix)-1 #height of matrix = number of items in first (1D) list
    width = graph.width = len(matrix[0]) # width = number of items in any given (2D) list, assuming matrix is rectangular
    for x in range (0,height):
        for y in range (0,width):
            value = matrix[x][y]
            if (value == 0): reward = 0 #empty space, no reward
            elif (value == 1): reward = -1 #mountain
            elif (value == 2): reward = -10 #wall, value not used
            elif (value == 3): reward = -2 #snake
            elif (value == 4): reward = 1 #barn
            elif (value == 50): reward = 50 #finish
            newNode = Node(y+1,height-x,int(value),int(reward)) # transform to standard x,y coordinates (1,1) in lower left
            graph.nodes.append(newNode) #add to the graph's node list
            if (newNode.location == (1,1)): #1,1 is start
                graph.start = newNode
            if (reward == 50): # node at (height,width) is the goal
                graph.goal = newNode
            #if (value==2): print newNode.location,newNode.reward

def printResults(graph): #from starting node, follow optimum path to goal
    print ""
    print "Optimal Path:"
    print "node location : node utility"
    tempNode = graph.start
    while (tempNode != None):
        print '{:^14s} {:^13.7f}'.format(tempNode.location,tempNode.utility)
        tempNode = tempNode.parent
    

def isFinished(graph,finished,gamma,epsilon,convergenceValue):
    finished = True
    length = len(graph.nodes)
    for index in range (0,length):
        delta = abs(graph.nodes[index].UValue - graph.nodes[index].UPrimeValue)
        graph.nodes[index].UValue = graph.nodes[index].UPrimeValue
        if (delta > convergenceValue): return False
        
    return finished

def main():
    if (len(sys.argv) < 3):
        print
        print "usage: python <filename.py>  <map filename.txt>  <epsilon value>"
        return
    worldMatrix = [] #initialize empty graph
    readInFile(worldMatrix) # read in matrix from file
    graph = Graph() #initialize empty graph
    buildWorld(graph,worldMatrix) #create nodes, populate graph with info from input file
    gamma = .9 # the 'decreasing return' factor
    epsilon = float(sys.argv[2])
    convergenceValue = epsilon*(1-gamma)/gamma
    finished = False
    while (finished == False):
        graph.updateExpectedUtilities(gamma)
        finished = isFinished(graph,finished,gamma,epsilon,convergenceValue)
        
    printResults(graph)
    
   




main()
