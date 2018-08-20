# Author: Donovan Guelde
# CSCI-3202 Fall 2015
# Assignment 6
# Implement Bayesian network

import getopt
import sys

class Graph(object):
	def __init__(self):
		self.children = []
		self.parents = []
		self.nodes = []
	
	def getNode(self,name):
		for index in range (0,len(self.nodes)):
			if self.nodes[index].name == name:
				return self.nodes[index]


class Node(object):
	def __init__(self,name):
		self.name = name
		self.parent = []
		self.children = []
		self.probability = {}
	def setProbability(self, key,value):
		del self.probability[key]
		self.probability[key] = value

def initNodes(graph):
	pollution = Node("P")
	graph.nodes.append(pollution)
	smoker = Node("S")
	graph.nodes.append(smoker)
	cancer = Node("C")
	cancer.children = [pollution,smoker]
	graph.nodes.append(cancer)
	xray = Node("X")
	graph.nodes.append(xray)
	dysn = Node("D")
	graph.nodes.append(dysn)
	pollution.probability["marginal"] = .9
	smoker.probability["marginal"] = .3
	cancer.probability["Ps"] = 0.05
	cancer.probability["PS"] = 0.02
	cancer.probability["ps"] = 0.03
	cancer.probability["pS"] = 0.001
	xray.probability["c"]=.9
	dysn.probability["c"]=.65

def getProb(event,graph):
	flag = False #flag is set if we need (1-P)
	if event.islower():
		event = event.upper()
		flag = True
	node = graph.getNode(event)
	if flag==True:
		if "marginal" in node.probability:
			return node.probability["marginal"]
		else: 
			totalProb = 0
			for key in node.probability:
				#print " inLoop"
				if len(key) > 1:
					key1 = key[0]
					#print "key1=",key1
					key2 = key[1]
					#print "key2=",key2
					totalProb = totalProb + node.probability[key]*getProb(key1,graph)*getProb(key2,graph)
			return totalProb #calculate probability recursively

	else:
		if "marginal" in node.probability:
			return 1-node.probability["marginal"]
		else: 
			totalProb = 0
			for key in node.probability:
				#print " inLoop"
				if len(key) > 1:
					key1 = key[0]
					#print "key1=",key1
					key2 = key[1]
					#print "key2=",key2
					totalProb = totalProb + node.probability[key]*getProb(key1,graph)*getProb(key2,graph)
			return 1-totalProb #calculate probability recursively


	return probability


def getMarginalProbability(variableList,graph):

	length = len(variableList)
	for index in range (0,length):
		print "marginal probability of",variableList[index],getProb(variableList[index],graph)

def getJointProbability(variableList,graph):
	length = len(variableList)
	jointProb = 1
	for index in range(0,length):
		jointProb = jointProb * (getProb(variableList[index],graph))
	print "Joint Probability of",variableList,jointProb

def getConditionalProbability(variableList,graph):
	event = variableList[0]
	conditional = variableList[1] #assumes P(a|b)


		




def main():
	flags = 'g:j:m:p:'
	commandString = getopt.getopt(sys.argv[1:],flags)
	print commandString
	optionFlag = commandString[0][0][0]
	argumentList = []
	
	print "optionFlag=",optionFlag
	variableList = commandString[0][0][1]
	print "variableList = ",variableList
	index = 0
	while index < len(variableList):
		variable = variableList[index]
		if (variable == "~"):
			index +=1
			variable = variable + variableList[index]
		index+=1
		print "variable = ",variable
		argumentList.append(variable)
	print len(argumentList)

	graph = Graph()
	initNodes(graph)
	cancer = graph.getNode("C")
	length = len(cancer.children)
	for index in range (0,length):
		print cancer.children[index].name,getProb(cancer.children[index].name,graph)
	for key in cancer.probability:
		print key, cancer.probability[key]
	
	#parse command
	if optionFlag == "-m":
		getMarginalProbability(variableList,graph)
	if optionFlag == "-j":
		getJointProbability(variableList,graph)
	if optionFlag == "-g":
		getConditionalProbability(variableList,graph)






main()