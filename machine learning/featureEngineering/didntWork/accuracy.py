# accuracy

import csv



trueLabel=[]
predictions=[]
text=[]
counter=0
firstTest=0
with (open("../data/spoilers/train.csv", 'rb')) as f:
	reader=csv.reader(f,delimiter=',')
	next(reader,None)
	for row in reader:
		counter+=1
		if (counter>10000):
			spoiler=row[1]
			sentence=row[0]
			trueLabel.append(spoiler)
			text.append(sentence)
			
f.close()


		

with (open("predictions.csv", 'rb')) as f:
	reader=csv.reader(f,delimiter=',')
	next(reader,None)
	for row in reader:
		guess=row[1]
		predictions.append(guess)
f.close

correct=0.
for index in xrange(len(trueLabel)):
	if(trueLabel[index]==predictions[index]):
		correct+=1
	#else:
	#	print trueLabel[index],text[index]
print "correct:",correct
print "accuracy:",correct/len(trueLabel)