__author__ = 'khandady'
import networkx as nx
import numpy as np
import random
import community
import matplotlib.pyplot as plt

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

#Get max degree
def MaxDegree(Q,vacnum):
    nodelist = Q.nodes()
    vaccinated = []
    sample = []
    sampledegree = []
    if(vacnum*2 >= 788):
        samplesize = 788
    else:
        samplesize = vacnum*2
    sample = random.sample(nodelist,samplesize)
    for s in sample:
        sampledegree.append(Q.degree(s))
    sarray = np.array(sample)
    sdarray = np.array(sampledegree)
    inds = sdarray.argsort()
    sortedsample = sarray[inds]
    for val in range(len(sortedsample)-1,len(sortedsample)-vacnum-1,-1):
        vaccinated.append(sortedsample[val])
    return vaccinated

def calculateModularity(network):
    bestPartition=community.best_partition(network)
    return community.modularity(bestPartition,network)

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

nodelist = N.nodes()
sucept = {}
for node in nodelist:
    sucept[node] = 0

modularityScore=[]
numberVaccinated=[]
vaccinatedSet=[]
modularityScore.append(calculateModularity(N))
numberVaccinated.append(0)

test = {}
#Max Degree Vaccinations
timesteparr = np.zeros(394)
sizearr = np.zeros(394)
xaxis = np.linspace(2,788,394)
vaccinated = []
for vacnum in range(1,390,1): #change back to 394
    vaccinated = MaxDegree(N,vacnum*2)
    test = sucept.copy()
    for vac in vaccinated:
        test[vac] = 2

    ##########################Calculate Vaccinated Modularity###############################
    nodea = []
    nodeb = []
    nodelist = []
    edgeweight = []

    f = open('Edgeweights.txt','r') #rebuild graph without vaccinated nodes
    for line in f:
        l = line.strip().split("\t")
        nodea.append(int(l[0]))
        nodeb.append(int(l[1]))
        edgeweight.append(int(l[2]))
    f.close()
    M = nx.Graph()
    for s in range(len(nodea)):
        if (nodea[s] not in vaccinated and nodeb[s] not in vaccinated):
            M.add_edge(nodea[s],nodeb[s],weight=edgeweight[s])
    try:
        modularityScore.append(calculateModularity(M))
        numberVaccinated.append(len(vaccinated))
        vaccinatedSet.append(vaccinated)
    except:
        pass
    print len(vaccinated)
    #########################################################################################

    
    for rep in range(30):#change back to 100 if success
        X = nx.Graph(N)
        vacstatus = test.copy()
        result=infection(X,vacstatus)
        #print(result)
        timesteparr[vacnum] += result[0]
        sizearr[vacnum] += result[1]
    
###################################save modularity results##################################
np.savetxt('MaxDegreevaclen.txt',timesteparr,fmt='%4.5e')
np.savetxt('MaxDegreevacsize.txt',sizearr,fmt='%4.5e')
with open("vaccinated.txt","w") as f:
    for item in vaccinatedSet:
        item=str(item)
        f.write(item)
        f.write('\n')
    f.close()
np.savetxt("numberVaccinated.txt",numberVaccinated,fmt='%i')
np.savetxt("modularity.txt",modularityScore,fmt='%10.5f')
plt.plot(numberVaccinated,modularityScore)
plt.xlabel("Number of Nodes Vaccinated")
plt.ylabel("Modularity Score")
plt.title("Modularity of Network as a Function of Number of Vaccinated Nodes")
plt.savefig("MaxDegreeModularity.pdf")

#Divide by number of runs for smoothing effect
timesteparr= np.true_divide(timesteparr,100)
sizearr = np.true_divide(sizearr,100)
