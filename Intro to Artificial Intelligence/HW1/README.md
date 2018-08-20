# Guelde_CSCI3202_Assignment1
CSCI-3202 HW 1

File can be executed via command line (python Guelde_Assignment1.py)  Class definitions are contained in ClassDefinitions.py,
and are imported by main program.  Upon execution, a series of automated tests are performed on the data structures.  If
desired, data structures can be maually checked/manipulated upon completion of automated tests.  If manual option is selected,
data structures are empty.

Notes on test structure:

  The Binary Search Tree used for testing is:         
	                      
                                                  1
	                                         2         3
	                                       4   5     6   7
	                                       8   9     10
	After node deletion:                          1
	                                         2         3
	                                       4   5     6   7
	                                           9
	                                           
	The graph is initialized with 10 nodes, integer values 1-10
	Graph edges are between the following pairs of nodes:
	      (1,7)(1,10)(2,8)(2,1)(3,9)(3,2)
				(4,10)(4,3)(5,1)(5,4)(6,2)(6,5)
				(7,3)(7,6)(8,4)(8,7)(9,5)(9,8)(10,6)(10,9)
				
Per class instructions, not all edge-cases have been accounted for, so it may be possible to force a run-time error.  Overall,
however, manual operation is generally functional.
