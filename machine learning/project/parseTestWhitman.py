import string


FILENAME="Whitman-clean.txt"

def main():
	titles=[]
	body=[]
	tempBody=[]
	with (open(FILENAME,'r')) as f:
		lines=f.readlines()
		start=False
		for line in lines:
			line=line.strip()
			for character in string.punctuation:
				line=line.replace(character,'')
			
			if (start==False): #skip all until first poem
				if line=="A HAPPY HOURS COMMAND":
					print "asdfas"
					titles.append(line)
					start=True
			if (start==True):
				if line.isupper(): #this is a title
					if line!="A HAPPY HOUR'S COMMAND":
						if(len(tempBody)>0):
							body.append(tempBody)
							titles.append(line)
							tempBody=[]
						else: 
							titles[-1] = titles[-1]+" "+ line
				else:
					if(len(line)>0):
						tempBody.append(line)
			
	body.append(tempBody)
	f.close()
	print len(titles)
	print len(body)
	for index in xrange(len(titles)):
		with (open("./poems/"+titles[index]+".txt",'w')) as f:
			f.write(str(body[index]))
		f.close




main()

