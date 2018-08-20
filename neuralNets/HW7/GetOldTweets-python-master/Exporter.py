# -*- coding: utf-8 -*-
import sys,getopt,datetime,codecs
if sys.version_info[0] < 3:
    import got
else:
    import got3 as got

def main(argv):
	'''
	if len(argv) == 0:
		print('You must pass some parameters. Use \"-h\" to help.')
		return

	if len(argv) == 1 and argv[0] == '-h':
		f = open('exporter_help_text.txt', 'r')
		print(f.read())
		f.close()

		return
	'''

	day=26
	day2 = 27
	month=2
	month2 = 2
	year2 = 2017
	year=2017

	for i in range(2247,2573):
		
		day+=1
		day2+=1
		if(month == 2 and day == 29):
			month = 3
			day = 1
		if (month == 9 and day == 31):
			month = 10
			day = 1
		if (month == 4 and day ==31):
			month = 5
			day = 1
		if (month == 6 and day == 31):
			month = 7
			day = 1
		if (month == 11 and day == 31):
			month = 12
			day = 1
		if (month == 12 and day == 32):
			month = 1
			day = 1
			year +=1
		elif(day == 32):
			month +=1
			day=1
		if(month2 == 2 and day2 == 29):
			month2 = 3
			day2 = 1
		if (month2 == 9 and day2 == 31):
			month2 = 10
			day2 = 1
		if (month2 == 4 and day2 ==31):
			month2 = 5
			day2 = 1
		if (month2 == 6 and day2 == 31):
			month2 = 7
			day2 = 1
		if (month2 == 11 and day2 == 31):
			month2 = 12
			day2 = 1
		if (month2 == 12 and day2 == 32):
			month2 = 1
			day2 = 1
			year2 +=1
		elif(day2 == 32):
			month2 +=1
			day2=1
		since = "{}-{}-{}".format(year,month,day);
		until = "{}-{}-{}".format(year2,month2,day2);
		#opts, args = getopt.getopt(argv, "", ("username=", "near=", "within=", "since=", "until=", "querysearch=", "toptweets", "maxtweets=", "output="))

		#tweetCriteria = got.manager.TweetCriteria()
		#outputFileName = "output_got.csv"
		tweetCriteria = got.manager.TweetCriteria().setMaxTweets(100).setQuerySearch('#bitcoin').setSince(since).setUntil(until).setTopTweets(True)
		
		#tweetCriteria.querySearch = "#bitcoin"
		#tweetCriteria.topTweets = True
		#tweetCriteria.maxTweets = 10
		outputFileName = "./topTweets/{}.csv".format(i)
		
		#tweets=got.manager.TweetManager.getTweets(tweetCriteria)
		'''
		for opt,arg in opts:
			if opt == '--username':
				tweetCriteria.username = arg

			elif opt == '--since':
				tweetCriteria.since = arg

			elif opt == '--until':
				tweetCriteria.until = arg

			elif opt == '--querysearch':
				tweetCriteria.querySearch = arg

			elif opt == '--toptweets':
				tweetCriteria.topTweets = True

			elif opt == '--maxtweets':
				tweetCriteria.maxTweets = int(arg)
			
			elif opt == '--near':
				tweetCriteria.near = '"' + arg + '"'
			
			elif opt == '--within':
				tweetCriteria.within = '"' + arg + '"'

			elif opt == '--within':
				tweetCriteria.within = '"' + arg + '"'

			elif opt == '--output':
				outputFileName = arg
		'''
		#here = os.path.dirname(os.path.realpath(__file__))
		#subdir = "tweets"
		#filepath = os.path.join(here, subdir, outputFileName)
		outputFile = codecs.open(outputFileName, "w+", "utf-8")

		outputFile.write('username,date,retweets,favorites,text,geo,mentions,hashtags,id,permalink')

		print('Searching...\n')

		def receiveBuffer(tweets):
			for t in tweets:
				outputFile.write(('\n%s,%s,%d,%d,"%s",%s,%s,%s,"%s",%s' % (t.username, t.date.strftime("%Y-%m-%d %H:%M"), t.retweets, t.favorites, t.text, t.geo, t.mentions, t.hashtags, t.id, t.permalink)))
			outputFile.flush();
			print('More %d saved on file...\n' % len(tweets))

		got.manager.TweetManager.getTweets(tweetCriteria, receiveBuffer)

	#except arg:
	#	print('Arguments parser error, try -h' + arg)
	#finally:
		outputFile.close()
		print('Done. Output file generated "%s".' % outputFileName)

if __name__ == '__main__':
	
	main(sys.argv[1:])
