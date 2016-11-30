from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import math 
import matplotlib.mlab as mlab
import random

spins =100
aveSpinGroup = 10
ising = []
J = -0.20 #alignmentE = -0.20 # energy of alignment

mu = 0.33
B = 0.
kB = 1.0
temperature = 100.
time = 150.
timeplots =time/1.0
timeplotsteps = int(time/timeplots)
energy = 0.0

def prob(r): # input a probability, gives you a 1 with that probability, or a 0
    x= 0.0
    x = random.uniform(0,1)
    if x <= r:
        return 1
    return 0
    
def clearIsing(isingT):  
    del isingT[:] 
    
def fillIsingCold(isingT):
    for i in xrange(spins):     #spin up is 1, spin down is -1
            isingT.append(-1)
   
def calcEnergy(isingTrial):
    firstTerm = 0.0
    secondTerm = 0.0
    for i in xrange(len(isingTrial)-1):
        firstTerm += isingTrial[i]*isingTrial[i+1]
    #print firstTerm
    firstTerm *= J
    for i in xrange(len(isingTrial)):
        secondTerm += isingTrial[i]
    firstTerm *= -B*mu    
    return (firstTerm +secondTerm)
def calcMag(isingTrial):
    magSum = 0.0
    for i in xrange(len(isingTrial)):
        magSum += isingTrial[i]
    return abs(magSum)
    
def flipNspins(n,T, isingI):
    flipped = []
    isingO = isingI[:]
    energy = calcEnergy(isingI) # original energy
    for i in xrange(n):
        flipped.append(int(random.uniform(0,1)*spins)) # choose a array of terms
    for i in xrange(len(flipped)): # flip those random terms in the output array
        isingO[flipped[i]] = -1 * isingO[flipped[i]]
    Etrial = calcEnergy(isingO) # new energy
    dE = float( Etrial-energy) # you keep it if the new energy is lower than the old
    #print "de is equal to %f" %dE 
   # print "dE is equal to %f" %dE
    if dE > 0.0: # if it is higher energy, then you don't automatically take it
      #  print "exp %f---------" %math.exp(-1.*dE/(kB*T))
        p = prob(math.exp(-1.*dE/(kB*T)))
        
        if p >0:  # with probablity (exp(-dE/ kbT)), keep the new arrangement
          a = 0
          #  print ""
         #   print "keep the new arrangment"
        else: #otherwise, flip everything back
            for i in xrange(len(flipped)):
                isingO[flipped[i]] *= -1
            #print "keep the old arrangement00000"
    count = 0
    for i in xrange(len(isingI)):
        if isingO[i]*isingI[i] == -1.:
            count = count +1
   # print "the sum of the differences is %d" %count
    return isingO

timeIsing = []
clearIsing(ising)
fillIsingCold(ising)

def runTimeIsing(timedIsing,temp, isingStart,time):
    timedIsing.append(isingStart)
    for i in xrange(int(time)):
        isingStart = flipNspins(1,temp, isingStart)
        timedIsing.append(isingStart)
    
print "hello"   
runTimeIsing(timeIsing, temperature,ising,time)
#print timeIsing
print "len ising %d" %len(timeIsing)
#print "len(timeIsing[1]) %d" %len(timeIsing[1])

def calcAveGroupSize(arr):
    groups =0
    i = 1
    while i< len(arr):# iterates through the total number of groups of same spins
        while arr[i] == arr[i-1]: # goes through a single group, to get you to the next one
            i+=1
            if i >= len(arr): # make sure you don't end of a group
                break
        groups +=1 # counts groups
        i+=1
    return (1.0*len(arr))/(1.* groups)

aveGroupSize =[]

def runIsingModelAveGroupSize(n,T, start,numSteps,trials):
    print "I'm heree2"
    for t in xrange(1, T): 
        iterator = start  
        summer = 0.0
        for trial in xrange(trials): 
            for i in xrange(numSteps): # run the ising model to a certain time
                iterator = flipNspins(1,t, iterator)
            if len(iterator) != 0:
                summer +=calcAveGroupSize(iterator)
                
        aveGroupSize.append((1.0*summer)/(1.0*trials))
    return aveGroupSize # returns an array of average group size, for each temperature till temperature T
coldStart = []
fillIsingCold(coldStart)

def runIsingModelKbTEnergy(n,T, start,numSteps,trials):
    aveGroupSize = []
    print "I'm heree2"
    for t in xrange(T): 
        iterator = start  
        summer = 0.0
        for trial in xrange(trials): 
            for i in xrange(numSteps): # run the ising model to a certain time
                iterator = flipNspins(1,t, iterator)
            if len(iterator) != 0:
                summer +=calcEnergy(iterator)
        aveGroupSize.append((1.0*summer)/(1.0*trials))
    return aveGroupSize # returns an array of average group size, for each temperature till temperature T
    
def runIsingModelCv(n,T, start,numSteps,trials):
    kB = 0.05
    Cv =[]
    Cval = 0.
    U2 = 0.
    U2summer = 0.
    U = 0.
    Usummer = 0.
    for t in xrange(1, T): 
        iterator = start  
        U2summer = 0.
        Usummer =0.
        for trial in xrange(trials): 
            for i in xrange(numSteps): # run the ising model to a certain time
                iterator = flipNspins(1,t, iterator)
            if len(iterator) != 0:
                Usummer +=calcEnergy(iterator)
                U2summer +=(calcEnergy(iterator)**2)
        U2 = (1.0*U2summer)/(1.0*trials)
        U = (1.0*Usummer)/(1.0*trials)
        Cval = (U2 - U**2)/(len(iterator)**2 *kB *t**2)
        Cv.append(Cval)
    return Cv # returns an array of average group size, for each temperature till temperature T

def runIsingModelTMag(n,T, start,numSteps,trials):
    print "I'm heree2"
    aveTMag = []
    ave = 0.0
    iterator = start
    for t in xrange(1, T):   
        summer = 0.0
        for trial in xrange(trials): 
            for i in xrange(numSteps): # run the ising model to a certain time
                iterator = flipNspins(1,t, iterator)
            if len(iterator) != 0:
                summer +=calcMag(iterator)
        ave = (1.0*summer)/(1.0*trials)
        print ave
        aveTMag.append(ave)
    return aveTMag # returns an array of average group size, for each temperature till temperature T
#x1 = np.arange(100) *kB
#aveGSize = runIsingModelAveGroupSize(spins, len(x1)+1, coldStart,100,5)
#fig2 = plt.figure(2) 
#plt.xlabel("kB * Temperature")
#plt.ylabel("Average Group Size")
#plt.title("Average Group Size versus Temperature")
#plt.plot(x1,aveGSize)
#fig2.show()

#x1 = np.arange(100) *kB
#aveTEnergy = runIsingModelKbTEnergy(spins, len(x1)+1, coldStart,200,40)
#fig2 = plt.figure(2) 
#plt.xlabel("Kb * Temperature")
#plt.ylabel("Average Energy")
#plt.title("Average Energy versus Temperature")
#plt.plot(x1, aveTEnergy)
#fig2.show()

#plot the average magnetization as a func of T
#x1 = np.arange(100) *kB
#aveTMag = runIsingModelTMag(spins*2, len(x1)+1, coldStart,100,5)
#fig2 = plt.figure(2) 
#plt.xlabel("Kb * Temperature")
#plt.ylabel("Average Magnetization")
#plt.title("Average Magnetization versus Temperature")
#plt.plot(x1,aveTMag)
#fig2.show()
# kb = 0.01

#This is to plot the heat capacity as a func of kb*T
#x1 = np.arange(100) *kB
#print x1
#print len(x1)
#aveTCv =runIsingModelCv(40 , len(x1)+1, coldStart,500, 20) # spins, top temp * kb, start,steps, trials averaged over
#fig2 = plt.figure(2) 
#plt.xlabel("Kb * Temperature")
#plt.ylabel("Heat Capacity")
#plt.title("Heat Capacity versus Temperature")
#print (len(aveTCv))
#plt.plot(x1,aveTCv )
#fig2.show()


fig1 = plt.figure(1)
scatter = []
scatterplot = []

#scatter1 = []
def averageSpin(array, i, spingroupsize):
    summed = 0.0
    for x in xrange(i,spingroupsize,1):
        summed+=array[x]
    print "ave is equale to %d" %summed
    return summed
    
    # this is to make the plots
for t in xrange(int(time)):
  #  del scatter1[:]
    for i in xrange(len(ising)-aveSpinGroup):
        if timeIsing[t][i] == -1 : #averageSpin(ising,i,aveSpinGroup):  # if it is spin down, then plot it(don't plot spin up) 
           # scatter1.append((t, i))
           scatter.append((t,i)) 
           if t%timeplotsteps ==0:
               scatterplot.append((t,i))
  #  scatter.append(scatter1)



print len(scatterplot)
print len(scatter)


diff = []
for i in xrange(len(timeIsing[0])):
    diff.append(timeIsing[0][i] - timeIsing[len(timeIsing)-1][i])
#sum(diff)
#print diff

plt.scatter([pt[0] for pt in scatterplot], [pt[1] for pt in scatterplot],  s=10, facecolor='0.0', lw = 0)
#plt.scatter(*zip(*scatter))
fig1.show()


