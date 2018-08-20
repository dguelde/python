import sys
import os

def main():
	#os.system('python file.py')
	os.system('python Exporter.py --querysearch \"#bitcoin\" --output \"test\"--maxtweets 10 --since 2011-01-01 --until 2011-01-02')