# Author: Donovan Guelde
# CSCI-3202 FAll 2015
# Assignment 8
# Hidden Markov Model
# Program will accept a CLA to use a different input file, if no CLA given, will use "typos20.data"
# output saved to test file "output.txt" 

import sys
from math import log
import math
import operator
import numpy

class HiddenMarkov(object): #a master class to contain all info
	def __init__(self): #each letter will have its own node to hold info
		self.nodes = []
		self.Null = letterNode("")
		self.letterCount=0
		self.wordCount=0
		self.observedWordList=[]
		self.stateWordList=[]

	def getNode(self,letter): #return node containing designated
		for index in range (0,len(self.nodes)):
			if self.nodes[index].letter == letter:
				return self.nodes[index]

	def initializeNodes(self): #manually initialize all nodes to save time checking if node exists later
		for index in range(97,123): #use ASCII values for a-z
			self.nodes.append(letterNode(chr(index)))

	def getMarginalProbability(self):
		#print "\nMarginal Probabilities\n"
		tempSum = 0
		outFile = open("output.txt","w")
		outFile.write ("\nMarginal Probabilities\n\n")
		for index in range(97,123): #use ASCII values for a-z, iterate through all letters
			letter = chr(index)
			currentNode = self.getNode(letter) #get node for current letter
			P = log(float(currentNode.numState)/float(self.letterCount)) #times this letter was in state/number of states
			#tempSum+=P
			#print "P({0}) = {1}".format((letter),(P))
			
			outFile.write ("P({0}) = {1}".format((letter),(P)))
			currentNode.probabilityInitial = P
			outFile.write("\n")
		outFile.close()
		#print tempSum #check to ensure sum of all probabilities = 1

	def getEmissionProbability(self): #display emission probabilities for all letters
		outFile = open("output.txt","a")
		outFile.write("\nEmission Probabilities:\n\n")
		for index in range(97,123): #use ASCII values for a-z, iterate through all letters
			letter = chr(index)
			#print "Emission Probabilities for",letter,":"
			tempSum=0 #a check value, make sure probabilities sum to 1.0
			currentNode = self.getNode(letter) #get node for current letter
			for index2 in range(97,123):#iterate through all letters (again)
				letter2 = chr(index2)
				timesObserved = currentNode.observed[letter2] #number of times letter2 is observed, given letter1 is in state
				P = log(float(timesObserved)/(currentNode.numState+26)) #calculate probability P(t|x), denominator + 26 for LaPlace smoothing
				#tempSum+=P
				#print "P ({0} | {1})={2}".format((letter2),(letter),(P))
				
				outFile.write("P ({0} | {1})={2}".format((letter2),(letter),(P)))
				outFile.write("\n")
				currentNode.emissionProbability[letter2] = P
		outFile.close()
			#print "	Sum of all probabilities for letter",letter,"=",tempSum #to ensure normConstant is correct
	
	def getTransitionProbability(self):
		outFile = open("output.txt","a")
		outFile.write("\nTransition Probabilities:\n\n")
		for index in range(97,123): #use ASCII values for a-z, iterate through all letters
			letter = chr(index)
			currentNode = self.getNode(letter)
			#print "Transission Probabilities for",letter,":"
			tempSum=0
			for index2 in range(97,123):#iterate through all letters (again)
				letter2 = chr(index2)
				previousNode=self.getNode(letter2)
				P = log(float(currentNode.next[letter2])/(float(currentNode.numState)+26))
				#tempSum+=P
				#print "P ( {0} | {1} )={2}".format((letter2),(letter),(P))
				
				outFile.write("P ( {0} | {1} )={2}".format((letter2),(letter),(P)))
				currentNode.transitionProbability[letter2] = P
				outFile.write("\n")
		outFile.close
			#print tempSum #checksum to ensure sum of all P = 1.0

	def viterbi(self):
		tempSum = 0.0 # to track overall error rate
		outFile = open("output.txt","a")
		score=0
		wrongScore=0
		matrix = numpy.zeros((26,26)) #a 26x26 matrix to hold calculation results
		#for index in range (0,len(self.stateWordList)): #iterate through every word
		for index in range (0,len(self.observedWordList)): #iterate through every word
			viterbiList=[]
			viterbiList2=[]
			observedWord = self.observedWordList[index] #run viterbi on this word
			stateWord=self.stateWordList[index]
			tempLetter=""
			tempMax=-9999
			mostLikelyWord=""
			observedLetter=observedWord[0]
			observedNode=self.getNode(observedLetter)
			
			#probability for first letter
			for index2 in range(0,26): #iterate through alphabet to compare probabilities
				possibleStateLetter = chr(97+index2) 
				stateNode=self.getNode(possibleStateLetter)
				viterbiList.append(stateNode.probabilityInitial+stateNode.emissionProbability[observedLetter]) #P(observedLetter|stateLetter)
			tempLetter=""
			tempMax=-99999
			mostLikelyWord+=chr(97+viterbiList.index(max(viterbiList))) #http://stackoverflow.com/questions/3989016/how-to-find-positions-of-the-list-maximum
			if (chr(97+viterbiList.index(max(viterbiList)))==stateWord[0]):
				score+=1

			for index2 in range(1,len(observedWord)): #iterate through the rest of the word
				score=0.0
				observedLetter=observedWord[index2]
				maxValue=0
				temp = 0
				for index3 in range(0,26):
					possibleStateLetter = chr(97+index2) 
					stateNode=self.getNode(possibleStateLetter)
				for index3 in range(0,26):
					currentLetter=chr(index3+97)
					currentStateNode=self.getNode(currentLetter)
					for index4 in range(0,26):
						previousLetter=chr(index4+97)
						previousStateNode=self.getNode(previousLetter)
						matrix[index3][index4] = previousStateNode.transitionProbability[currentLetter]+currentStateNode.emissionProbability[observedLetter]+viterbiList[index4]
				for index3 in range(0,26):
					tempValue=-999999
					for index4 in range(0,26):
						if matrix[index3][index4]>tempValue:
							tempValue=matrix[index3][index4]
					viterbiList[index3]=tempValue

				tempValue=-99999
				tempX=0
				tempY=0
				for index5 in range(0,26):
					for index6 in range(0,26):
						if matrix[index5][index6]>tempValue:
							tempX = index5
							tempY = index6
							tempValue=matrix[index5][index6]
				mostLikelyWord+=chr(tempX+97)

			outFile.write("State Sequence = {0}\n".format((stateWord)))
			outFile.write("Observed Sequence = {0}\n".format((observedWord)))

			outFile.write("Viterbi Output = {0}\n".format(mostLikelyWord))
			outFile.write("x[t] ={0}\n".format((math.exp(tempValue))))
			for index9 in range(0,len(stateWord)):
				if stateWord[index9]==mostLikelyWord[index9]:
					score+=1
					tempSum+=1

			outFile.write("Error Rate = {0}\n".format((1-score/len(stateWord))))
		outFile.write("Overall Error Rate = {0}\n".format((1-tempSum/float(self.letterCount))))

		outFile.close

			#for index2 in range(1,len(observedWord)): #check remaining letters in current word


class letterNode(object):
	def __init__(self,letter):
		self.letter = letter #leter
		self.numObserved = 0 # number times letter is observed, regardless of state
		self.numState = 0 # number times letter is in state, regardless of observed
		self.stateAndObserved = 0 #number of times letter observed while also in state

		self.probabilityInitial = 0



						#this records what letter is observed when this node's letter is in state
		self.observed = {'a':1,'b':1,'c':1,'d':1,'e':1,'f':1,'g':1,'h':1 
						,'i':1,'j':1,'k':1,'l':1,'m':1,'n':1,'o':1,'p':1
						,'q':1,'r':1,'s':1,'t':1,'u':1,'v':1,'w':1,'x':1
						,'y':1,'z':1} #all values start at 1 for Laplace smoothing

		self.emissionProbability = {'a':1,'b':1,'c':1,'d':1,'e':1,'f':1,'g':1,'h':1 
						,'i':1,'j':1,'k':1,'l':1,'m':1,'n':1,'o':1,'p':1
						,'q':1,'r':1,'s':1,'t':1,'u':1,'v':1,'w':1,'x':1
						,'y':1,'z':1}

						#what is in state when this letter is observed 
		self.stateIfObserved = {'a':1,'b':1,'c':1,'d':1,'e':1,'f':1,'g':1,'h':1 
						,'i':1,'j':1,'k':1,'l':1,'m':1,'n':1,'o':1,'p':1
						,'q':1,'r':1,'s':1,'t':1,'u':1,'v':1,'w':1,'x':1
						,'y':1,'z':1} #all values start at 1 for Laplace smoothing

		self.transitionProbability = {'a':1,'b':1,'c':1,'d':1,'e':1,'f':1,'g':1,'h':1 
						,'i':1,'j':1,'k':1,'l':1,'m':1,'n':1,'o':1,'p':1
						,'q':1,'r':1,'s':1,'t':1,'u':1,'v':1,'w':1,'x':1
						,'y':1,'z':1}

						#next state (this state is s in p(s'|s))
		self.next = {'a':1,'b':1,'c':1,'d':1,'e':1,'f':1,'g':1,'h':1
						,'i':1,'j':1,'k':1,'l':1,'m':1,'n':1,'o':1,'p':1
						,'q':1,'r':1,'s':1,'t':1,'u':1,'v':1,'w':1,'x':1
						,'y':1,'z':1} #all values start at 1 for Laplace smoothing


def readFile(HMM):
	inFile = open("typos20.data","r") #if no CLA provided, use text file for part 1
	if (len(sys.argv) > 1): #if CLA provided, open that file
		try:
			inFile = open(sys.argv[1],"r")
		except IOError:
			print sys.argv[1],"not found"
			sys.exit(1)
	previousLetter = "?"
	stateLetter = "?"
	currentObservedWord = [] #store each letter until hit new word, save in markov.wordList
	currentStateWord = []
	if (inFile):
		
		for line in inFile:
			line = line.rstrip() #remove \n
			if line[0] == "_": #new word, no previous letter, nothing to do
				HMM.observedWordList.append(currentObservedWord)
				HMM.stateWordList.append(currentStateWord)
				previousLetter = "?"
				HMM.wordCount+=1
				currentStateWord=[]
				currentObservedWord=[]
				continue
			previousLetter = stateLetter #remember last state to update "next" dictionary 
			observedLetter = line[2] #observed letter
			stateLetter = line[0] #state letter
			currentObservedWord += observedLetter
			currentStateWord += stateLetter
			observedNode = HMM.getNode(observedLetter) #get node corresponding to observed letter
			observedNode.numObserved +=1  #increment numObserved counter of observed letter
			stateNode = HMM.getNode(stateLetter) #may be same as observed node (probably is)
			stateNode.numState+=1 #increment state counter of state letter
			if stateNode==observedNode: #if observed == state
				stateNode.stateAndObserved+=1 #counter to track number of times observed == state
			stateNode.observed[observedLetter]+=1 #for P(t|x), remember what letter is observed while in state
			observedNode.stateIfObserved[stateLetter]+=1 #what letter is in state when this letter observed
			if (previousLetter != "?"): #if this is not the first letter of a new word
				HMM.getNode(previousLetter).next[stateLetter] +=1 #update 'next' dict of previous letter
			HMM.letterCount+=1
	else: print "Cannot open file"




def main():
	HMM = HiddenMarkov() #declare empty model
	HMM.initializeNodes() #initialize all nodes/letters
	readFile(HMM) #populate HMM/nodes with info from file

	HMM.getMarginalProbability() #print marginal/initial probabilities
	HMM.getEmissionProbability() #print emission probabilities
	HMM.getTransitionProbability() #print transition probabilities
	
	HMM.viterbi()

	
	

main()
