# -*- coding: utf-8 -*-
from pylab import *
import numpy as np
import math 
import random
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

vmin = 0.
vmax = 1.0
ME = 2.7182818284590452354 # Euler ’s const
def f1(x): # The integrand
    return np.cos(x)

def f2(x):
    return exp(-x)
def derf1(x):
    return -np.sin(x)
def derf2(x):
    return exp(x)


def integrateTrapRule(func,N, xo, xf):
    summer = func(xo)
    h = (xf-xo)/(N-1)
    xi = xo
    for i in xrange(2,N):
        print ("i %d" %i)
        xi +=h
        summer += h* func(i*xi)
    summer+= (func(xf)*h/2)
    return summer
    
def integrateSimpsonRule(func,N,xo, xf):
    summer = func(xo)
    h = (xf-xo)/(N-1.)
    wi = 0.
    xi = xo
    for i in xrange(1,N):
        xi +=h
        if i %3 ==0:
            wi=(h/3.)
        if i %3 ==1:
            wi=(4.*h/3.)
        if i %3 ==2:
            wi=(2.*h/3.)
        summer += wi*func(xi*i)
    return summer
def integrateGaussianRule(func,xo, xf,N):
    return 1
    

print( "Trap Rule Error")
error = (integrateTrapRule(f2,2, vmin,vmax) - (1.-1./ME))/((1.-1./ME))
print integrateTrapRule(f2,3, vmin,vmax)
print "actual error of 2 %f" %error
print (1.-1./ME)
print integrateTrapRule(f2,10, vmin,vmax)
error10 = (integrateTrapRule(f2, 100,vmin,vmax) - (1.-1./ME))/((1.-1./ME))
print "actual error of 10 %f" %error10

x1 = np.arange(2,80)
y1 = []
trapy1 =[]
simpy1 =[]
for i in xrange(len(x1)):
    trapy1.append(log10((integrateTrapRule(f2,x1[i], vmin,vmax) - (1.-1./ME))/((1.-1./ME))))
    simpy1.append(log10((integrateSimpsonRule(f2,x1[i], vmin,vmax) - (1.-1./ME))/((1.-1./ME))))
    #y1.append((integrateTrapRule(f2, x1[i],vmin,vmax)))
fig1 = plt.figure(1)

plt.plot(x1,trapy1, label = "trapezoid error")
plt.plot(x1,simpy1, label = " simpson's error")
plt.xlabel("number of terms")
plt.ylabel("Error of Integral Value")
legend(bbox_to_anchor=(0.65, .85), loc=2, borderaxespad=0.)
plt.title("log 10 plot of Error of Trapezoidal and Simpson integration error for e^x")
fig1.show()

print( "Simpsons Rule Error")
error = (integrateSimpsonRule(f2,2, vmin,vmax) - (1.-1./ME))/((1.-1./ME))
print integrateSimpsonRule(f2,3, vmin,vmax)
print "actual error of 2 %f" %error
print (1.-1./ME)
print integrateSimpsonRule(f2,10, vmin,vmax)
error10 = (integrateSimpsonRule(f2, 100,vmin,vmax) - (1.-1./ME))/((1.-1./ME))
print "actual error of 10 %f" %error10


print log10(1.-(1./ME))
