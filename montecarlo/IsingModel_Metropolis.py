from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import math 
import matplotlib.mlab as mlab
import random

spins =100
aveSpinGroup = 10
ising = []
J = 0.20 #alignmentE = -0.20 # energy of alignment

mu = 0.33
B = 1.
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
    
def fillIsing(isingT):
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
fillIsing(ising)
#print "========="
#print ising
#print "====-----===="
timeIsing.append(ising)
for i in xrange(int(time)):
    ising = flipNspins(1,temperature, ising)
    timeIsing.append(ising)

print "len ising %d" %len(timeIsing)
print "len(timeIsing[1]) %d" %len(timeIsing[1])


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


