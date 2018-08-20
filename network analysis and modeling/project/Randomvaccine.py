__author__ = 'khandady'
import networkx as nx
import numpy as np
import random

#Build Network

#Simulate infection using SEIR model. Two time steps for exposed to become infected. Two time steps before infected becomes recovered.
#Recovered have no contact with rest of network. Infection can only spread in the day.
def infection(X,status):
    nodelist = X.nodes()
    #Choose random starting node for infection
    start = random.choice(nodelist)
    infected = 1
    exposed = 0
    newinfarr = [start]
    exposedlist = []
    recovered = []
    timestep = 0
    #Continue until no more infected or exposed people
    while infected != 0 or exposed != 0:
        timestep += 1
        if(timestep % 2 == 1):
            for vec in newinfarr:
                for neighbor in X.neighbors(vec):
                    if(status[neighbor] == 0):
                        if((1-(1-.003)**X[vec][neighbor]['weight']) >= random.random()):
                            exposedlist.append(neighbor)
                            status[neighbor] = 1
                            exposed += 1
                newinfarr.remove(vec)
                recovered.append(vec)
        else:
            newinfarr = exposedlist
            exposedlist = []
            infected = exposed
            exposed = 0
    return [timestep, len(recovered)]

#Generate Graph
nodea = []
nodeb = []
nodelist = []
edgeweight = []
#read in network and data
f = open('Edgeweights.txt','r')
for line in f:
    l = line.strip().split("\t")
    nodea.append(int(l[0]))
    nodeb.append(int(l[1]))
    edgeweight.append(int(l[2]))
f.close()

#Create network
N = nx.Graph()
for s in range(len(nodea)):
    N.add_edge(nodea[s],nodeb[s],weight=edgeweight[s])

#print(N[366][379])
#print(N[379][366])
nodelist = N.nodes()
sucept = {}
for node in nodelist:
    sucept[node] = 0

test = {}

#Random Vaccinations

timesteparr = np.zeros(395)
sizearr = np.zeros(395)
xaxis = np.linspace(0,788,395)
vaccinated = []
for vacnum in range(0,394,1):
    vaccinated = random.sample(nodelist,vacnum*2)
    test = sucept.copy()
    for vac in vaccinated:
        test[vac] = 2
    for rep in range(100):
        X = nx.Graph(N)
        vacstatus = test.copy()
        result=infection(X,vacstatus)
        timesteparr[vacnum] += result[0]
        sizearr[vacnum] += result[1]

#Divide by number of runs for smoothing effect
timesteparr= np.true_divide(timesteparr,100)
sizearr = np.true_divide(sizearr,100)

np.savetxt('Randomvaclen.txt',timesteparr,fmt='%4.5e')
np.savetxt('Randomvacsize.txt',sizearr,fmt='%4.5e')