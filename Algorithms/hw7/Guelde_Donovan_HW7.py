# Author: Donovan Guelde
# CSCI-3104 Fall 2015
# Assignment 6
# cloth cutting

#"On my honor, as a University of Colorado at Boulder student, I have neither given nor received unauthorized assistance."


import sys

class Graph(object):
	def __init__(self,productArray,profitMap):
		self.productArray = productArray
		self.profitMap = profitMap
		self.path = []
		self.numberCuts=0

	def profit(self,x,y): #determice profit value of a rectangle XxY
		if (self.profitMap[x][y][0] > -1): #if a value already exists for these dimensions, just return it
			return self.profitMap[x][y][0]
		if (x==0 or y==0): return 0 #too small to hold any products
		highestUnitProfit = 0 #tracks profit/unit
		tempMax = 0 #tracks profit
		tempX = 0 #X,Y location of next cut
		tempY = 0
		
		for index in range (0,len(self.productArray)): #compare our x,y values to base cases, stored in the product array
			if (x>=self.productArray[index][0] and y>=self.productArray[index][1]):
				if highestUnitProfit < self.productArray[index][3]: #checks for highest unit profit rather than highest profit, to maximize profit/space
					
					highestUnitProfit = self.productArray[index][3]
					tempMax = self.productArray[index][2] #gets actual profit value from product info array
					tempX = self.productArray[index][0] #x and y values of possible cut locations
					tempY = self.productArray[index][1]
		
			
		if tempMax > self.profitMap[x][y][0]:
			self.profitMap[x][y][0] = tempMax #update profit map location if we found a higher value
		choiceA = self.profit(tempX,y-tempY)+self.profit(x-tempX,y) #profit of making a vertical cut
		choiceB = self.profit(x-tempX,tempY)+self.profit(x,y-tempY) #profit of making a horizontal cut
		
		
		#horizontal or vertical cut?
		if choiceA > choiceB:
			self.numberCuts+=1
			self.path.append([self.numberCuts,x,y,1,tempX])
		if choiceB > choiceA:
			self.numberCuts+=1
			self.path.append([(self.numberCuts,x,y,0,tempY)])

		total = max(choiceA,choiceB) + tempMax
		return total





def readFile(X,Y,n,productArray):
	textFile = sys.argv[1]
	inFile = open(textFile)
	lineNumber = 1 #line number counter
	for line in inFile:
		line = line.rstrip("\n")
		if line == '':continue #skip empty lines

		if lineNumber == 1: #line #1 gives dimensions of cloth
			temp = line.split(" ")
			X = int(temp[0])
			Y = int(temp[1])
		if lineNumber == 2: #number of possible products
			n = line
		if lineNumber>=3: #line 3+ give product info
			temp = line.split(" ") 
		
			productArray.append([]) #add new empty list to array
			for index in range (0,len(temp)): 
				if temp[index] == '':continue
				productArray[lineNumber-3].append(int(temp[index])) #place product info in list, piece by piece
			productArray[lineNumber-3].append(float(productArray[lineNumber-3][2])/float((productArray[lineNumber-3][0])*float(productArray[lineNumber-3][1]))) #add an entry for 'unit value'
			
		lineNumber+=1
	inFile.close()	
	return (int(X),int(Y),n,productArray)




	

def main():
	X = Y = n = 0 #X,Y = starting dimensions of cloth, n = # of products
	productArray = [] #x dimension,y dimension, profit, unit value (x*y/profit)
	profitMap = [] #store total profit
	results = []
	(X,Y,n,productArray) = readFile(X,Y,n,productArray) #get X,Y,n,product info from text file

	for index in range (0,X+1):
		profitMap.append([])
		for index2 in range (0,Y+1):
			profitMap[index].append([-1,0]) #initialize profit map 

	for index in range(0,len(productArray)):#make a 'profit map' to quickly compare coordinates with known profit values
		tempX = productArray[index][0]
		tempY = productArray[index][1]
		tempProfit = productArray[index][2] #profit from this product
		tempUnitProfit = productArray[index][3]
		profitMap[tempX][tempY] = [tempProfit,tempUnitProfit] #tuple containing both profit and unit profit


	graph = Graph(productArray,profitMap) #initialize graph object to hold info (not really a graph, just used to using the name 'graph')
	
	profit = graph.profit(X,Y) #run the algorithm
	
	outFile = open(sys.argv[2],"w")
	length = len(graph.path)
	
	
	outFile.write(str(profit)+ " " + str(length)+ '\n')

	for index in range(length-1,0,-1):
		output = str(graph.path[index])
		outFile.write(output[1:-1] +'\n')
	outFile.close()

	
	

main()



