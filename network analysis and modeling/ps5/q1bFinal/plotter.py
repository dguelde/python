import matplotlib.pyplot as plt


def readFile(fileName):
	data = []
	with (open(fileName,'r')) as f:
		for line in f:
			data.append(float(line))
	f.close()
	return data

def main():
	commonNeighbor=readFile('commonNeighborsHVR_550accuracy.txt')
	degreeProduct=readFile('degreeProductHVR_550accuracy.txt')
	shortestPath=readFile('shortestPathHVR_550accuracy.txt')
	fValues=readFile('predictEdgesHVR_550fValues.txt')

	plt.plot(fValues,degreeProduct)
	plt.plot(fValues,commonNeighbor)
	plt.plot(fValues,shortestPath)
	plt.legend(['Degree Product','Common Neighbor','Shortest Path'],loc=4)
	plt.xlabel('f')
	plt.ylabel('AUC')
	plt.savefig('./HVR5Malaria.png')
	plt.show()

main()