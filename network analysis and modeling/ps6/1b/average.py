with open('1b_100.txt','r')as f:
	counter=0
	tempSumFirst=0
	tempSumLast=0
	for line in f:
		if(counter<100000):
			tempSumFirst+=float(line)
		if(counter>100000):
			tempSumLast+=float(line)
		counter+=1
	print "first:",tempSumFirst/100000
	print "last:",tempSumLast/100000

