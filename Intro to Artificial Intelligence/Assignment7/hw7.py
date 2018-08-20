# Author: Donovan Guelde
# CSCI-3202 Fall 2015
# Assignment 7

import csv


def readfile(samples): #reads sample info from text file, puts it in a list
	infile = open("samples.txt","r")
	if (infile):
		counter = 0
		reader = csv.reader(infile,delimiter = ",")
		for value in reader:
			length = len(value)
			for index in range(0,length):
				samples.append(float(value[index]))
		
		infile.close()

	else: print "Cannot open file"

def applySamplestoEvents(samples,events):
	for index in range(0,25): #distribute sample values over events, assumes 100 sample values (100/4 = 25)
		events[0].append(samples[index*4+0])
		events[1].append(samples[index*4+1])
		events[2].append(samples[index*4+2])
		events[3].append(samples[index*4+3])
	return events


def runPriorSamples(events): #take sample info from list, use it to generate 25 individual, complete trials
	sampleSpace = [] #will be a 2D list containing true/false values for cloudy, sprinkler, rain, wet 
	for index in range (1,26): #calculate cloudy=true/false
		if events[0][index] <=.5:
			sampleSpace.append([True])
		else: sampleSpace.append([False])

	for index in range(0, len(sampleSpace)): #sprinkler true/false
		if sampleSpace[index][0]==True: #if cloudy
			if (events[1][index+1]<=.1): #event array is offset by one from sampleSpace array, so index+1
				sampleSpace[index].append([True]) #sprinkler
			else: sampleSpace[index].append([False])
			if (events[2][index+1]<=.8): #rain
				sampleSpace[index].append([True])
			else: sampleSpace[index].append([False])
		else: #if not cloudy
			if events[1][index+1]<=.5: #sprinkler
				sampleSpace[index].append([True])
			else:
				sampleSpace[index].append([False])
			if events[2][index+1]<=.2: #rain
				sampleSpace[index].append([True])
			else:
				sampleSpace[index].append([False])

	for index in range (0,25): #wet = true/false
		if (sampleSpace[index][1]==[True] and sampleSpace[index][2] == [True]): #sprinkler = true, rain = true
			if events[3][index+1] <=.99:
				sampleSpace[index].append([True]) #wet = true
			else:sampleSpace[index].append([False])
		if (sampleSpace[index][1]==[True] and sampleSpace[index][2] == [False]): #sprinkler true, rain false
			if events[3][index+1] <=.90:
				sampleSpace[index].append([True]) #wet = true
			else:sampleSpace[index].append([False])
		if (sampleSpace[index][1]==[False] and sampleSpace[index][2] == [True]): #sprinkler false, rain true
			if events[3][index+1] <=.90:
				sampleSpace[index].append([True]) #wet = true
			else:sampleSpace[index].append([False])
		if (sampleSpace[index][1]==[False] and sampleSpace[index][2] == [False]): #sprinkler false, rain false
			if events[3][index+1] <=0.0:
				sampleSpace[index].append([True]) #wet = true
			else:sampleSpace[index].append([False])
	return sampleSpace #we now have a list of 25 complete trials


def displayPriorSampleResults(sampleSpace):
	#calculate, display P values
	cloudyCounter = 0.0
	rainCounter = 0.0
	cloudGivenRain = 0.0
	wetCounter = 0.0
	sprinklerGivenWet = 0.0
	cloudyAndWet = 0.0
	sprinklerGivenCloudyAndWet = 0.0
	rainCounter2 = 0.0
	for index in range (0,25): #iterate through sampleSpace, totalling relevent events
		if sampleSpace[index][0] == True:
			cloudyCounter+=1 # cloudy = true
			if sampleSpace[index][3] == [True]:
				cloudyAndWet +=1 #cloudy = true, wet = true
				if sampleSpace[index][1] == [True]:
					sprinklerGivenCloudyAndWet+=1 #sprinkler true given cloudy and wet
		if sampleSpace[index][2] == [True]:
			rainCounter+=1 #rain true
			if sampleSpace[index][0] == True:
				cloudGivenRain+=1 #cloudy given rain
		if sampleSpace[index][3] == [True]:
			wetCounter +=1 #wet true
			if sampleSpace[index][1] == [True]:
				sprinklerGivenWet+=1 #sprinkler true given wet
		if sampleSpace[index][2] == [True]: rainCounter2+=1
	print ""
	print "Prior Sampling Results:"
	print "		P(c=true) =",cloudyCounter/25
	print "		P(c=true|rain=true) =",cloudGivenRain/rainCounter
	print "		P(s=true|w=true) =",sprinklerGivenWet/wetCounter
	print "		P(s=true|c=true,w=true) =",sprinklerGivenCloudyAndWet/cloudyAndWet
	print "		P(r=true) =",rainCounter2/25
	print "		P(w=true) =",wetCounter/25

def displayRejectionSampleResults(samples):
	length = len(samples) #length of sample list
	# P(c=true), sample from the top, no priors
	cloudyCounter = 0.0
	for index in range (0,length):
		if samples[index] <=0.5: #cloudy true
			cloudyCounter+=1
	print ""
	print "Rejection Sampling Results: "
	print "		P(c=true) =",cloudyCounter/float(index)

	# P(c=true|rain=true); p(rain=true) = 0.5 (.8*.5 + .2*.5)
	rainCounter = 0.0
	cloudyCounter = 0.0
	for index in range(0,length-1): #-1 because we (may) use 2 samples per iteration
		if samples[index] <= .5: #rain = true
			rainCounter+=1
			index+=1
			if samples[index]<=.5: #P(c=true) = 0.5
				cloudyCounter+=1
	print"		P(c=true|rain=true) =",cloudyCounter/rainCounter

	#P(s=true|w=true), P(w = true) = .5985, P(s=true) = .3
	wetCounter = 0.0
	sprinklerCounter = 0.0
	for index in range(0,length-1): #-1 because we (may) use 2 samples per iteration
		if samples[index] <= .5985: #P(wet=true) = true
			wetCounter+=1
			index+=1
			if samples[index]<=.3: #P(s=true) = 0.3
				sprinklerCounter+=1
	print"		P(s=true|w=true) =",sprinklerCounter/wetCounter
	#P(s=true|c=true,w=true)
	wetAndCloudyCounter = 0.0
	sprinklerCounter = 0.0
	for index in range(0,length-2): #-2 because we may have to use 3 samples per iteration
		if (samples[index]<=.5): #c=true
			index+=1
			if samples[index]<=.5985: #both wet and cloudy = true
				wetAndCloudyCounter+=1
				index+=1
				if samples[index]<=.3:
					sprinklerCounter+=1
	print"		P(s=true|c=true,w=true) =",sprinklerCounter/wetAndCloudyCounter





def main():
	samples = []
	events = [["cloudy"],["sprinkler"],["rain"],["wet"]]#events[0] = [cloudy], events[1] = [sprinkler], etc...
	readfile(samples)
	events = applySamplestoEvents(samples,events)
	sampleSpace = runPriorSamples(events)
	displayPriorSampleResults(sampleSpace)
	displayRejectionSampleResults(samples)




main()