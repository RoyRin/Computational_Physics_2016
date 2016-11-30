from pylab import *
import numpy as np
import math 
import random
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy.optimize as optimization

rate= 0.1
n0 = 200.
time = 40
t1 = np.arange(0,time,1)

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
plt.ylabel("value")
plt.title("radioactive decay")
fig2.show()



fig3 = plt.figure(3)

plt.plot(t1,log10(N), label='Number')
plt.plot(t1,log10(dN), label ='rate')

legend(bbox_to_anchor=(0.65, .85), loc=2, borderaxespad=0.)
plt.xlabel("time")
plt.ylabel("Log (base 10)")
plt.title("log 10 of Radioactive Decay")
fig3.show()

A = np.vstack([t1, np.ones(len(t1))]).T

def func(a,b,x):
    return a*x +b

m, c = np.linalg.lstsq(A, log10(N))[0]

print "slope and offset"
print(m, c)
print"sadsadadsada"
#print optimization.leastsq(func(), t1, N)

fig4 = plt.figure(4)

plt.plot(t1,log10(N), label='Number')

plt.plot(t1,func(m,c,t1), label ='least squares line')

legend(bbox_to_anchor=(0.65, .85), loc=2, borderaxespad=0.)
plt.xlabel("time")
plt.ylabel("Log (base 10)")
plt.title("log 10 of Radioactive Decay versus least squares line")
fig4.show()

