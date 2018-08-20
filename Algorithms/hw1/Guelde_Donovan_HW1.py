# Author:  Donovan Guelde
# CSCI-3104 Fall 2015
# write a python program to generate and sort random numbers
#
# On my honor, as a University of Colorado at Boulder student,
# I have neither given nor received unauthorized assistance.
#
# time info from: http://www.dreamincode.net/forums/topic/278225-calculating-the-runtime-of-a-function/

from random import randint
import time
import gc

def main():


	n = 0;
	
	timeList = [] #declare list to store time info
	gc.disable
	
	while (n<1):
		n = input("Input a positive integer: ")
		timeList[:] = [] #clear time info
		for x in range (0,10):
			my_list = [] #initialize empty list to store random #s
			generateNumbers(my_list,n) 
			selectSort(my_list,n,timeList)
		#find min, max, and average sort times
		min = timeList[0]
		max = timeList[0]
		sum = 0
		for x in range (0,10):
			if timeList[x] < min:
				min = timeList[x]
			if timeList[x] > max:
				max = timeList[x]
			sum = sum + timeList[x]
		print "min = ",min
		print "max = ",max
		print "average time = ",sum/10
		n = 0

def generateNumbers(my_list,n): #generate n random integers (0-10n)
	for y in range (0,n): #repeat n times
		my_list.append(randint(0,10*n))


def selectSort(my_list,n,timeList):
	#selection sort of numbers just generated
	startTime = time.time() #begin timer
	for startIndex in range (0,n): #outer loop runs n-times
		startValue = my_list[startIndex]
		switchFound = False
		for comparisonIndex in range (startIndex+1, n): #inner loop runs (n-starting index) times
			if my_list[comparisonIndex] <= startValue: #found a value to switch
				switchFoundAt = comparisonIndex #index of switch
				startValue = my_list[comparisonIndex] #value of switch
				switchFound = True #switch flag
		if switchFound == True:
			my_list[switchFoundAt], my_list[startIndex] = my_list[startIndex], my_list[switchFoundAt] #swap values
	elapsedTime = time.time() - startTime #stop timer
	timeList.append(elapsedTime) #save elapsed time in list
	



main()

