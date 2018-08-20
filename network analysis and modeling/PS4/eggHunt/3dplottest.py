from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show


def readFiles():
	global L
	global S
	e = []
	xPrime = 0.
	for x in range(0,161):
		xPrime = float("{0:.3g}".format(xPrime))
		e.append(xPrime)
		xPrime+=0.1


	p = []
	xPrime=0.0
	for x in range(0,100):
		xPrime = float("{0:.3g}".format(xPrime))
		p.append(xPrime)
		xPrime+=0.01

	L=[]
	S=[]
	
	counter=0
	for index in range(0,len(e)):
		ePrime = e[index]
		filename = 'E{}length.txt'.format(ePrime)
		
		with (open(filename,'r')) as f:
			L.append([])
			lines=f.readlines()
			for line in lines:
				L[counter].append(float(line))
		counter+=1
	f.close()

	counter=0
	for index in range(0,len(e)):
		ePrime = e[index]
		filename = 'E{}size.txt'.format(ePrime)
		with (open(filename,'r')) as f:
			S.append([])
			lines=f.readlines()
			for line in lines:
				S[counter].append(float(line))
		counter+=1
	f.close()


	#L=np.array(L)
	return p,e



def main():
	P,E = readFiles()

	X,Y = meshgrid(P,E)
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	surf = ax.plot_surface(X, Y, L, rstride=1, cstride=1, cmap=cm.terrain,linewidth=0, antialiased=False)
	ax.set_xlabel("P")
	ax.set_ylabel("Epsilon")
	ax.set_zlabel("Length")
	fig.colorbar(surf, shrink=0.5, aspect=5)
	#plt.zlabel("Epidemic Length")
	plt.show()
	plt.close()

	fig = plt.figure()
	ax = fig.gca(projection='3d')
	surf = ax.plot_surface(X, Y, S, rstride=1, cstride=1, cmap=cm.RdBu,linewidth=0, antialiased=False)
	plt.show()
	plt.close()	

main()
