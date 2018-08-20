# Donovan Guelde
#CSCI 5454 PS4

import sys
import numpy as np
import random
import time
import math


MATRIXPRESET=0
CURSOR_UP_ONE = '\x1b[1A' #http://stackoverflow.com/questions/12586601/remove-last-stdout-line-in-python
ERASE_LINE = '\x1b[2K'

class Game:
	def __init__(self,fileName=None):
		self.gameType=self.getGameType()
		if fileName==None:
			self.matrix = self.createMatrix()
		else:
			self.matrix = self.readMatrix(fileName)
		self.n = input("number of rounds: ")
		self.player1 = AI(self.matrix,1,self.n)
		self.player2 = AI(self.matrix,2,self.n)

	def getGameType(self):
		print "Choose game"
		print "1: player vs AI"
		print "2: AI vs AI"
		return int(input("Enter 1 or 2:"))

	def readMatrix(self,fileName):
		tempMatrix=[]
		with(open(fileName,'r')) as f:
			for line in f:
				line = line.split()
				tempLine=[]
				for item in line:
					tempLine.append(float(item))
				tempMatrix.append(tempLine)
		matrix=np.asarray(tempMatrix)
		return matrix


	def createMatrix(self):
		if (MATRIXPRESET==0):
			lowerBound,upperBound = input("enter bounds for matrix values, seperated by a comma (ex: 1,5): ")
			lowerBound=int(lowerBound)
			if lowerBound<0:
				lowerBound-=1
			upperBound=int(upperBound)
			spread=(upperBound-lowerBound)
			if(lowerBound<=0):
				spread+=1
			matrix=np.add(np.multiply(np.random.random((5,5)),spread),lowerBound).astype(int)
			matrix = np.random.uniform(lowerBound,upperBound+1,(5,5)).astype(int)
			matrix=matrix.astype(float)
		if(MATRIXPRESET==1): #rock paper scissors
			matrix=np.array([[1,0,2],[2,1,0],[0,2,1]]) #rock paper scissors
		if(MATRIXPRESET==2):
			matrix=np.transpose(np.array([[5,2,3,4,1],[4,3,4,5,1],[2,4,5,6,1],[2,3,4,5,1],[3,2,6,9,1]]).astype(float))
		return matrix

	


	def game_value(self):
		matrixMin,matrixMax=np.amin(self.matrix),np.amax(self.matrix)
		lossRange=(matrixMax-matrixMin)


		T=10000
		n=0.
		colPlayer = AI(self.matrix,1,self.n)
		rowPlayer = AI(self.matrix,2,self.n)
		maxLoss=(np.amax(np.absolute(self.matrix)))
		eta = (-1*math.sqrt(math.log(float(len(self.matrix)))/T))
		colPlayer.weightMultiplier=eta
		rowPlayer.weightMultiplier=eta
		for index in range (0,T-1):
			#n+=1.
			i=colPlayer.AImove()
			j=rowPlayer.AImove()
			colPlayer.updateWeights(j)
			rowPlayer.updateWeights(i)

		gameValue=np.dot(np.dot(rowPlayer.weights,self.matrix),colPlayer.weights)
		np.set_printoptions(linewidth=90)
		print "Column Player Weights:"
		print colPlayer.weights
		print "Row Player Weights:"
		print rowPlayer.weights
		print "Game Value:",gameValue
		print "Expected Score:",gameValue*self.n,"\n"
		return gameValue*self.n




	def playRound(self):
		#print self.game_value()
		player1Score,player2Score=0,0
		player1Wins=0
		print "\nResults of game_value():"
		gameValue=self.game_value()
		print "Payoff Matrix:"
		print self.matrix.astype(int)
		print "\n"
		for index in xrange(self.n):
			if(self.gameType==1):
				while True:
					try:
						i=int(input("input move: "))
						print(CURSOR_UP_ONE + ERASE_LINE + CURSOR_UP_ONE)
						break
					except SyntaxError:
						print "try again"
				while (i>len(self.matrix)-1):
					i=int(input("Out Of Bounds, input move: "))
			else:
				i=self.player1.AImove()
			j=self.player2.AImove()
			roundResult=self.matrix[j][i]
			if(roundResult>=np.median(self.matrix)):
				player1Wins+=1
			player1Score+=roundResult
			player2Score-=roundResult
			if(self.gameType==2):
				#print "update"
				self.player1.updateWeights(j)
			self.player2.updateWeights(i)
			#print "column played:",i,"  Row Played:",j,"Payoff:",roundResult
		
		
		print "Column Player Score:",player1Score
		print "Column Player Expected Score:",gameValue
		
		

class AI:
	def __init__(self,matrix,playerNumber,nRounds):
		self.m = 0
		self.playerNumber=playerNumber
		self.weightMultiplier=-np.sqrt(np.log(float(len(matrix)))/nRounds)
		self.matrix = self.setMatrix(matrix,playerNumber)
		self.weights=self.initializeWeights() #initialize weight verctor to (1/m,1/m,...,1/m)
		self.j=0
		self.accumulatedLossVector = np.zeros(self.m)
		

	def initializeWeights(self):
		weights = np.empty(self.m)
		if(np.sum(self.matrix)==0):
			for index in xrange(len(self.matrix)):
				weights[index]=1./(len(self.matrix))
		else:
			minimum = float(np.min(self.matrix))
			tempMatrix = np.add(self.matrix,-1*minimum)	
			for index in xrange(len(weights)):
				weights[index]=np.sum(tempMatrix[index])/np.sum(tempMatrix)
		return weights

	def AImove(self):
		self.j=np.random.choice(self.m,replace=True,p=self.weights)
		return self.j

	def updateWeights(self,i):
		badResults=np.zeros(len(self.matrix))
		lossVector=np.transpose(self.matrix)[i]	
		if self.playerNumber==1:
			loss=np.absolute(np.subtract(lossVector,np.amax(lossVector))) #col player maximize
		else:
			loss = np.subtract(lossVector,np.amin(lossVector)) #row player minimize
		if np.array_equal(loss,badResults): loss=np.ones(len(self.matrix)) #if all elements in loss vector are zero, set all elements to 1
		loss = np.divide(loss,np.sum(loss)) #regularize loss to [0,1]
		self.accumulatedLossVector=np.add(self.accumulatedLossVector,loss) #update sum in weight term
		temp=np.multiply(self.accumulatedLossVector,self.weightMultiplier)
		weights = np.exp(temp)
		weights = np.divide(weights,np.sum(weights))
		self.weights = weights

	def setMatrix(self,matrix,playerNumber):
		
		if playerNumber==2:
			temp = matrix
		else:
			temp = np.transpose(matrix)
		self.m = len(temp)
		return temp

def main():
	if len(sys.argv)==1:
		game = Game()
	else:
		game=Game(sys.argv[1])
	game.playRound()



main()
