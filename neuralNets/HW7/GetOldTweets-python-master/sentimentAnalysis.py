from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def main():
	outputFile = "tweetsPositiveNegativeTopTweets.txt"
	analyzer = SentimentIntensityAnalyzer()
	outfile = open(outputFile,'w')
	for i in range(2573):
		
		fileName = "./topTweets/{}.csv".format(i)
		with open(fileName,"r") as f:
			positive=0
			negative=0
			lines = f.readlines()[1:]
			for line in lines:
				tokens = line.split(',')
				text = tokens[4]
				vs = analyzer.polarity_scores(text)
				if(vs['compound']>0):
					positive+=1
				if(vs['compound']<0):
					negative+=1
			outfile.write("{},{}\n".format(positive,negative))
			f.close()
	outfile.close()

main()