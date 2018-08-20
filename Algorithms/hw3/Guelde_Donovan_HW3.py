# Author: Donovan Guelde
# CSCI-3104 Fall 2015
# HW3
# apply maximum subarray to a set of data
# data file specified by command line argument
# max subarrays adapted directly from CLRS pseudocode
#
# On my honor, as a University of Colorado at Boulder student, 
# I have neither given nor received unauthorized assistance.

import sys
import csv
from decimal import Decimal
from math import floor

#fills a list of price changes, and a list of corresponding dates, from the CSV file
def readFile(historicalData,dateList):
	date = 0
	prevDate = 0
	price = 0
	previousPrice = 0
	firstDayValueRead = False # a flag to make the first value in the price list the change drom day 1 to day 2, rather than the day 1 price
	dataFile = sys.argv[1]
	inFile = open(dataFile,'r')
	inFile.next() #skip the first 2 lines of the CSV file
	inFile.next()
	csv_inFile = csv.reader(inFile)
	for row in csv_inFile:
		if (firstDayValueRead):
			previousPrice = price
			price = float(row[3]) #convert from string to float
			dateList.append(row[0])
			historicalData.append(round(previousPrice-price,2)) #round the float to 2 digits right of the decimal
		else:
			price = float(row[3])
			firstDayValueRead = True
	dateList.reverse() #csv file in in reverse chronological order, so reverse the lists
	historicalData.reverse()

def findMaxCrossingSubarray(A,low,mid,high):
	maxLeft = maxRight = 0
	leftSum = -sys.maxint - 1
	total = 0
	for i in range (int(mid), int(low),-1):
		total +=A[i]
		if total>leftSum:
			leftSum = total
			maxLeft = i
	rightSum = -sys.maxint - 1
	total = 0
	for j in range (int(mid) + 1, int(high)):
		total += A[j]
		if total > rightSum:
			rightSum = total
			maxRight = j
	return (maxLeft, maxRight, leftSum + rightSum)

def findMaxSubArray(A,low,high):
	if high == low: #base case: low == high
		return (low,high,A[int(low-1)])
	else:  #recursive calls
		 mid = floor((low+high)/2)
		 (leftLow,leftHigh,leftSum) = findMaxSubArray(A,low,mid) 
		 (rightLow,rightHigh,rightSum) = findMaxSubArray(A,mid+1,high)
		 (crossLow,crossHigh,crossSum) = findMaxCrossingSubarray(A,low,mid,high)
		 if (leftSum >= rightSum and leftSum >= crossSum):
		 	return (leftLow,leftHigh,leftSum)
		 elif (rightSum >= leftSum and rightSum >= crossSum):
		 	return (rightLow,rightHigh,rightSum)
		 else:
		 	return (crossLow,crossHigh,crossSum)

def kadaneAlgorithm(A,length): # O(n) algorithm to do the same job, Kadane's Algorithm, https://en.wikipedia.org/wiki/Maximum_subarray_problem
	best = 0
	current = 0
	for i in range (0,length):
		current += A[i]
		if (current<0):
			current = 0
		best = max(best,current)
	return best

def main():
	historicalData = []
	dateList = []
	readFile(historicalData,dateList) #use 2 lists so I don't have to mess with tuples
	length = len(historicalData)
	
	(buyDate,sellDate,profit) = findMaxSubArray(historicalData,0,length)
	print "Buy on:",dateList[buyDate],"Sell on:",dateList[sellDate+1],"Profit:",profit
	print "algorithm 2:",kadaneAlgorithm(historicalData,length)
	

main()
