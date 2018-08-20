import string

FILENAME="dickinson-clean.txt"

def main():
	titles=[]
	body=[]
	tempBody=[]
	counter=0
	with (open(FILENAME,'r')) as f:
		lines=f.readlines()
		start=False
		for line in lines:
			line=line.strip() #strip leading/trailing whitespace
			for character in string.punctuation: #strip punctuation
				line=line.replace(character,'')
			
			if (start==False): #skip all until first poem
				if line=="SUCCESS":
					titles.append(counter) #not many titles...
					start=True
			if (start==True):
				if line.isupper(): #this is a title or divider
					if line!="SUCCESS":
						if(len(tempBody)>0):
							body.append(tempBody)
							titles.append(line)
							tempBody=[]
						
				else:
					if(len(line)>0):
						tempBody.append(line)
			counter+=1
	body.append(tempBody)
	f.close()
	print len(titles)
	print len(body)
	for index in xrange(len(titles)):
		with (open("./dickinson/"+str(index)+".txt",'w')) as f:
			f.write(str(body[index]))
		f.close




main()

