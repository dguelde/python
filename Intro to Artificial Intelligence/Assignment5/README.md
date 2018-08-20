This program can be executed via the command line using command line arguments for the text file (world information) and 
  espilon value.  Ex. <$python Assignment5.py WORLD1MDP.txt 0.5> executes the program, using WORLD1MDP.txt as the source
  to build the 'world' and epsilon = 0.5


I tested this program with epsilon = 10,1,.5, and .1.  The resulting optimal path did not change for any value epsilon,
  nor did the utility of the nodes along that path, to 7 decimal places.  This seemed strange to me, so I ran the program
  once more, epsilon = 100, and the resulting optimal path was the same, but some of the utility values were slightly
  different (largest difference was the utility of the beginning node, with a difference of .0006.)
