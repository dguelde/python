from Guelde_Assignment1 import Stack
from Guelde_Assignment1 import Node
from Guelde_Assignment1 import BinaryTree
from Guelde_Assignment1 import Graph
from Guelde_Assignment1 import manualOperation
import Queue



def dequeueTest(q):
	print "Beginning Queue Test"
	print "Enqueueing 1,2,3,4,5,6,7,8,9,10"
	q.put(1)
	q.put(2)
	q.put(3)
	q.put(4)
	q.put(5)
	q.put(6)
	q.put(7)
	q.put(8)
	q.put(9)
	q.put(10)
	print "1,2,3,4,5,6,7,8,9,10 enqueued"
	print "Dequeuing(FIFO): "
	print "Dequeued 1 = ",q.get() == 1
	print "Deququed 2 = ",q.get() == 2
	print "Deququed 3 = ",q.get() == 3
	print "Deququed 4 = ",q.get() == 4
	print "Deququed 5 = ",q.get() == 5
	print "Deququed 6 = ",q.get() == 6
	print "Deququed 7 = ",q.get() == 7
	print "Deququed 8 = ",q.get() == 8
	print "Deququed 9 = ",q.get() == 9
	print "Deququed 10 = ",q.get() == 10
	raw_input ("Press Enter key to continue tests")
	print " "
	
	

def stackTest(stack):
	print "Beginning Stack Test"
	print "Pushing 1,2,3,4,5,6,7,8,9,10 onto stack"
	stack.push(1)
	stack.push(2)
	stack.push(3)
	stack.push(4)
	stack.push(5)
	stack.push(6)
	stack.push(7)
	stack.push(8)
	stack.push(9)
	stack.push(10)
	print "1,2,3,4,5,6,7,8,9,10 pushed onto stack"
	print "Popping (LIFO): "
	print stack.pop()
	print stack.pop()
	print stack.pop()
	print stack.pop()
	print stack.pop()
	print stack.pop()
	print stack.pop()
	print stack.pop()
	print stack.pop()
	print stack.pop()
	raw_input ("Press Enter key to continue tests")
	print " "

def treeTest(tree):
	print "Beginning Binary Search Tree Test"
	tree.addRoot(1)
	tree.add(2,1)
	tree.add(3,1)
	tree.add(4,2)
	tree.add(5,2)
	tree.add(6,3)
	tree.add(7,3)
	tree.add(8,4)
	tree.add(9,5)
	tree.add(10,6)
	#tree populated:          1
	#                    2         3
	#                  4   5     6   7
	#                  8   9    10 
	tree.printTree(tree.root)
	print "deleting nodes 8 and 10"
	tree.delete(10)
	tree.delete(8)
	#resulting tree:          1
	#                    2         3
	#                  4   5     6   7
	#                      9
	tree.printTree(tree.root)
	print "see comments for tree geometry"
	raw_input ("Press Enter key to continue tests")
	print " "

'''
graph testing:
graph consists of vertexes: 1,2,3,4,5,6,7,8,9,10
				  edges: (1,7)(1,10)(2,8)(2,1)(3,9)(3,2)
				  		 (4,10)(4,3)(5,1)(5,4)(6,2)(6,5)
				  		 (7,3)(7,6)(8,4)(8,7)(9,5)(9,8)(10,6)(10,9)
'''
def graphTest(graph):
	graph.addVertex(1)
	graph.addVertex(2)
	graph.addVertex(3)
	graph.addVertex(4)
	graph.addVertex(5)
	graph.addVertex(6)
	graph.addVertex(7)
	graph.addVertex(8)
	graph.addVertex(9)
	graph.addVertex(10)
	print "10 vertexes added (1,2,3,4,5,6,7,8,9,10)"
	graph.addEdge(1,7)
	graph.addEdge(1,10)
	graph.addEdge(2,8)
	graph.addEdge(2,1)
	graph.addEdge(3,9)
	graph.addEdge(3,2)
	graph.addEdge(4,10)
	graph.addEdge(4,3)
	graph.addEdge(5,1)
	graph.addEdge(5,4)
	graph.addEdge(6,2)
	graph.addEdge(6,5)
	graph.addEdge(7,3)
	graph.addEdge(7,6)
	graph.addEdge(8,4)
	graph.addEdge(8,7)
	graph.addEdge(9,5)
	graph.addEdge(9,8)
	graph.addEdge(10,6)
	graph.addEdge(10,9)
	print "20 edges added"
	graph.findVertex(1)
	graph.findVertex(3)
	graph.findVertex(5)
	graph.findVertex(7)
	graph.findVertex(9)


def main():
	

	#manualOperation()
	# initialize data structures
	q = Queue.Queue() 
	stack = Stack() 
	tree = BinaryTree()
	graph = Graph()
	#queue testing
	dequeueTest(q)
	stackTest(stack)
	treeTest(tree)
	graphTest(graph)
	menuCommand = "a"
	menuCommand = raw_input ("Enter 1 to manually manipulate data structures, any other key to exit\n")
	if (menuCommand == "1"):
		manualOperation()


	



main()





