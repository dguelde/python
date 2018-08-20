# Author: Donovan Guelde
# CSCI-3202 Fall 2015
# Prof. Hoenigman
# HW 1 Python data structures
#Stack implementation adapted from: http://stackoverflow.com/questions/4688859/stack-data-structure-in-python
#graph implementation adapted from: http://www.python-course.eu/graphs_python.php
#       and:https://www.safaribooksonline.com/library/view/python-cookbook/0596001673/ch01s06.html
#general reference: https://docs.python.org/2/tutorial/datastructures.html


#Definition of Data Structures/Classes/Functions

# implement queue module
import Queue




class Stack(list): #implement stack class using a linked list
	def __init__(self):
		self.__stack = []
	def push(self, p):
		self.__stack.append(p)
	def checkSize(self):
		return len(self.__stack)
	def pop(self):
		return self.__stack.pop()

class Node(object): #define Node class for use in binary tree
	def __init__(self, key_int):
		self.key = key_int
		self.left = None
		self.right = None
		self.parent = None

class BinaryTree(Node): #define binary tree characteristics/functions
	def __init__(self): #declare root (empty at this point)
		self.root = None
		self.Null = Node(None) #declare null node
		self.Null.right = self.Null
		self.Null.left = self.Null
	def addRoot(self,value): #declare root if empty
		self.root = Node(value)
		self.root.right = self.Null
		self.root.left = self.Null
	def add(self, value, parentValue): #add new node, if called for
		parentNode = self.search(parentValue) #find node with parent value
		if parentNode == self.Null:
			print "Parent not found"
		elif (parentNode.left != self.Null and parentNode.right != self.Null):
			print "Parent has two children, node not added"
		else: #if parent node found and 0 or 1 children, add new child node
			newNode = Node(value)
			newNode.parent = parentNode
			newNode.left = self.Null
			newNode.right = self.Null
			if parentNode.left==self.Null: #add left child if empty
				parentNode.left=newNode
			else:
				parentNode.right=newNode #add right child if left is already full
	def delete(self,value): #find and delete node w/ specified value
		tempNode = self.search(value) #find node
		if tempNode == self.Null:
			print "Node not found"
		elif tempNode.left != self.Null or tempNode.right != self.Null:
				print "node not deleted, has children"
		else:
			if self.root == tempNode:
				self.root = None
			elif tempNode.parent.left == tempNode: 
				tempNode.parent.left = self.Null
			elif tempNode.parent.right == tempNode:
				tempNode.parent.right = self.Null
			del tempNode
	def printTree(self, n):
		if (self.root!= None):
			if n!= self.Null:
				print n.key
				self.printTree(n.left)
				self.printTree(n.right)
		else:
			print "Tree is empty"
	def search(self,value): #compares specified value to root
		if self.root.key == value:
			return self.root
		else:
			n = self.searchTree(self.root,value) #continues search if not found @ root
			return n
	def searchTree(self,n,value):
		m = n #need to duplicate so both left/right sides can be checked seperately, since left/right children are not sorted
		if n.key == value:
			return n
		if n.left != self.Null:
			n = self.searchTree(n.left,value)
			if n.key == value:
				return n
		if m.right != self.Null:
			m = self.searchTree(m.right,value)
			if m.key == value:
				return m
		return self.Null #returns null if value not found
		
class Graph(object):
	def __init__(self, dictionary={}):
		self.dictionary = dictionary
	def deleteGraph(self):
		self.dictionary.clear()
	def addVertex(self,value):
		if value in self.dictionary:
			print "Vertex already exists"
		else:
			self.dictionary[value] = []
	def addEdge(self,value1,value2):
		if value1 not in self.dictionary or value2 not in self.dictionary:
			print "One or more vertices not found"
		else: #adds 1 to 2, and 2 to 1 (in dictionary)
			self.dictionary[value1].append(value2)
			self.dictionary[value2].append(value1)
	def findVertex(self, value):
		if value in self.dictionary:
			print value, ": ", self.dictionary[value]
		else:
			print "Vertex not in dictionary"

def manualOperation():
	q = Queue.Queue() 
	stack = Stack() 
	tree = BinaryTree()
	graph = Graph()
	menuCommand_int = -1
	while menuCommand_int!=0:
		print "Main Menu"
		print "1.  Queue"
		print "2.  Stack"
		print "3.  Binary Tree"
		print "4.  Graph"
		print "0.  Quit"
		menuCommand_int = input("Select option: ")
		while menuCommand_int == 1:
			print "Queue Menu"
			print "1.  Queue value"
			print "2.  Dequeue value"
			print "0.  Return to Main Menu"
			queueMenuCommand_int = input("Select option: ")
			if queueMenuCommand_int == 1:
				inputValue_int = input("enter value to queue: ")
				q.put(inputValue_int)
			if queueMenuCommand_int == 2:
				if q.qsize()==0:
					print "Queue is empty"
				else:
					dequeuedValue_int = q.get()
					print "Dequeued value: %d" %dequeuedValue_int
			if queueMenuCommand_int == 0:
				break
		while menuCommand_int == 2:
			print "Stack menu"
			print "1.  Push"
			print "2.  Pop"
			print "3.  Check size"
			print "0.  Return to Main Menu"
			stackMenuCommand_int = input("Select option: ")
			if stackMenuCommand_int == 1:
				valueToPush_int = input("integer to push: ")
				stack.push(valueToPush_int)
			if stackMenuCommand_int == 2:
				if stack.checkSize() == 0:
					print "stack is empty"
				else:
					valueFromPop = stack.pop()
					print valueFromPop, " returned from stack"
			if stackMenuCommand_int == 3:
				print "Stack size: ",stack.checkSize()
			if stackMenuCommand_int == 0:
				break
		while menuCommand_int == 3:
			print "Binary Tree menu"
			print "1.  Add node"
			print "2.  Delete node"
			print "3.  Print node values"
			print "0.  Return to Main Menu"
			treeMenuCommand_int = input("Select option: ")
			if treeMenuCommand_int == 1:
				valueToAdd_int = input("enter value to add: ")
				if tree.root == None:
					tree.addRoot(valueToAdd_int) #declare root if empty
				else:
					parentValue_int = input("Enter key value of parent node: ")
					tree.add(valueToAdd_int,parentValue_int)
			if treeMenuCommand_int == 2:
				if (tree.root == None):
					print "Tree is empty"
				else:
					valueToDelete_int = input("enter value to delete: ")
					tree.delete(valueToDelete_int)
			if treeMenuCommand_int == 3:
				tree.printTree(tree.root)
			if treeMenuCommand_int == 0:
				break
		while menuCommand_int == 4:
			print "Graph Menu"
			print "1.  Add Vertex"
			print "2.  Add Edge"
			print "3.  Find Vertex"
			print "0.  Return to Main Menu"
			graphMenuCommand_int = input("Select option: ")
			if graphMenuCommand_int == 1:
				value = input("Enter Vertex Value: ")
				graph.addVertex(value)
			if graphMenuCommand_int == 2:
				value1 = input("Enter First Vertex Value: ")
				value2 = input("Enter Second Vertex Value: ")
				graph.addEdge(value1,value2)
			if graphMenuCommand_int == 3:
				value = input("Enter Vertex Value: ")
				graph.findVertex(value)
			if graphMenuCommand_int == 0:
				break
	print "Goodbye"

