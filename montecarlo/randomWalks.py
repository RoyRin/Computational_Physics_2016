# -*- coding: utf-8 -*-
from pylab import *
import numpy as np
import math 
import random
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

random.seed(None)
steps = 1000

datax1 = []
datay1 = []
datax2 = []
datay2 = []
datax3 = []
datay3 = []
datax4 = []
datay4 = []

stepsize = 1.
a = 0.0
b= 0.0
x= 0.0
y = 0.0
def randomWalk(n):
    walk =[]
    x= 0.
    y= 0.
    for i in xrange(n):
        a =(random.uniform(-1.,1.)) * stepsize # −1 =< x =< 1   
        b = (random.uniform(-1.,1.) ) * stepsize # −1 =< y =< 1
        L = math.sqrt(a**2 + b**2)
        x += a/L
        y +=b/L
        walk.append([x,y])
    return walk

distA = []
dist = 0.0
trials = 100
sumDist = 0.0
for d in xrange(1,steps,10):
    dist = 0.0
    for i in xrange(trials):
        dist = randomWalk(d)
        sumDist = dist[len(dist)-1][0]**2 + dist[len(dist)-1][1]**2
    distA.append(sumDist/trials)

fig2 = plt.figure(2)
plot(distA)
fig2.show()

fig1 = plt.figure(1)
plots = 3
for i in xrange(plots):
    plt.plot(*zip(*randomWalk(steps)))
plt.xlabel("X - distance of random walks")
plt.ylabel("Y - distance of random walks")
plt.ylim([-math.sqrt(steps),math.sqrt(steps)])
plt.xlim([-math.sqrt(steps),math.sqrt(steps)])
plt.title("%d " %plots + "random walks in %d steps" %steps)
plt.grid(True)
fig1.show()