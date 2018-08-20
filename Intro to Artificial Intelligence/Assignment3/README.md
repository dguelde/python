CSCI-3202 Assignment 3

Usage: $python Guelde_Assignment3.py \<file containing world information\> \<"1" or "2" (heuristic choice, without quotes)\>

Heuristic choice 1 runs the standard "Manhattan distance" as discussed in class and lecture materials.
Heuristic choice 2 uses a straight line distance and average movement cost to estimate movement cost to goal.
          The distance is calculated by finding the x and y distance to goal (graph max height - node y value)
          and (graph max width - node x value), then using the Pythagorian equation (x^2 + y^2 = z^2)
          to generate a straight line distance to goal.  
          The average movement cost is calculated by counting up all the squares, totaling the individual movement costs,
          and dividing (sum movement costs / total nodes = average movement cost)
          Approximate movement cost to goal = straight line distance * average movement cost
          
Heuristic #2 was chosen because it represents the most simplistic way to get from one node to another.  As movement penalties for 
  traversing mountains are built in to the heuristic, this should represent the average cost to move from one node to another.
  
Performance between the two heuristics is identical for the provided world.txt files:
Pathfinding Results for World1.txt heuristic 1
path: (1, 1) -> (2, 1) -> (3, 1) -> (4, 2) -> (4, 3) -> (4, 4) -> (5, 5) -> (6, 5) -> (7, 5) -> (8, 6) -> (8, 7) -> (9, 8) -> (10, 8)
12 nodes open
50 nodes closed

Pathfinding Results for World1.txt heuristic 2
path: (1, 1) -> (2, 1) -> (3, 1) -> (4, 2) -> (4, 3) -> (4, 4) -> (5, 5) -> (6, 5) -> (7, 5) -> (8, 6) -> (8, 7) -> (9, 8) -> (10, 8)
12 nodes open
50 nodes closed

Pathfinding Results for World2.txt heuristic 1
path: (1, 1) -> (2, 1) -> (3, 1) -> (4, 2) -> (4, 3) -> (5, 4) -> (6, 5) -> (7, 5) -> (8, 6) -> (9, 7) -> (10, 8)
12 nodes open
48 nodes closed

Pathfinding Results for World2.txt heuristic 2
path: (1, 1) -> (2, 1) -> (3, 1) -> (4, 2) -> (4, 3) -> (5, 4) -> (6, 5) -> (7, 5) -> (8, 6) -> (9, 7) -> (10, 8)
12 nodes open
48 nodes closed
