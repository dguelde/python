from csv import DictReader, DictWriter


train = list(DictReader(open("../data/spoilers/train.csv", 'r')))
pageCounter=0
tropeCounter=0
pages=[]
tropes=[]


for entry in train:
	spoiler=0
	page=entry['page']
	trope=entry['trope']
	print entry['spoiler']
	if entry['spoiler']=='True': spoiler=1
	print spoiler,"\n"
	if page not in pages:
		pages.append(page)
	else:
		index=pages.index(page)

	if trope not in tropes:
		tropes.append(trope)

pagesDictionary=dict.fromkeys(pages)
tropesDictionary=dict.fromkeys(pages)




print 
print len(pages)
print len(tropes)
