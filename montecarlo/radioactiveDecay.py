from pylab import *
import numpy as np
import math 
import random
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


rate = 0.01
time = 1000
t1 = np.arange(0,time,1)
print(t1)
n0  = 10000.
def pRate(n):
    return -rate * n;
def contDecay(rate1):
    return n0 *np.exp(-t1 *rate1)
contD = contDecay(rate)
fig1 = plt.figure(1)
plt.plot(t1,contD)
fig1.show()

def prob(r): # input a probability, gives you a 1 with that probability, or a 0
    x= 0.0
    x = random.uniform(0,1)
    if x <= r:
        return 1
    return 0

def discreteDecay(N,dN,rate1):
    N.append(int(n0))
    derN = 0.0
    for x in xrange(N[0]):
            derN += prob(rate1)
    dN.append(derN)     
    for i in xrange(1,len(t1)):
        derN =0
        for x in xrange(N[i-1]):
            derN += prob(rate1)
        N.append(N[i-1] - derN )
        dN.append(derN)
    
N = []
dN = []
discreteDecay(N,dN,rate)

fig2 = plt.figure(2)
#plt.semilogy(t1,N, label='Number')
plt.plot(t1,N, label='Number')
plt.plot(t1,dN, label ='rate')

legend(bbox_to_anchor=(0.65, .85), loc=2, borderaxespad=0.)
plt.xlabel("time")
plt.ylabel("Log (base 10)")
plt.title("radioactive decay")
fig2.show()

fig3 = plt.figure(3)
#plt.semilogy(t1,N, label='Number')
plt.semilogy(t1,N, label='Number')
plt.semilogy(t1,dN, label ='rate')

legend(bbox_to_anchor=(0.65, .85), loc=2, borderaxespad=0.)
plt.xlabel("time")
plt.ylabel("Log (base 10)")
plt.title("radioactive decay")
fig3.show()
       