# -*- coding: utf-8 -*-
from numpy import *
from sys import version
from pylab import *
import numpy as np
import math 
import random
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

max_in = 11 # Numb intervals
vmin = 0.
vmax = 1. # Int ranges
ME = 2.7182818284590452354 # Euler ’s const
w = zeros ( (2001) , float)
x = zeros ( (2001) , float)

def f1(x): # The integrand
    return np.cos(x)

def f2(x):
    return exp(-x)
def derf1(x):
    return -np.sin(x)
def derf2(x):
    return exp(x)



def gauss(npts, job , a, b, x, w) : # num of points, job
    # job 1 = trapezoid , job 2 = simpson, job 3 = gaussian quadrature integration
    m = i = j = t = t1 = pp = p1 = p2 = p3 = 0.
    eps = 3.*10**-14 # Accuracy : * * * * * *ADJUST THIS * * * * * * * !
    m = int ((npts + 1)/2 )
    for i in range (1, m + 1):
        t = cos(math.pi*(float(i)-0.25)/(float(npts)+0.5))
        t1 = 1
        while((abs(t-t1))>= eps) :
            p1 = 1.
            p2 = 0.
            for j in range (1,npts+1) :
                p3 = p2
                p2 = p1
                p1 = (( 2.*float(j)-1)*t*p2-(float(j)-1.)*p3)/(float(j))
            pp = npts *( t * p1-p2) /( t * t-1. )
            t1 = t 
            t = t1-p1 / pp
        x[i-1] =-t
        x[npts -i ] = t
        w[i-1] = 2./((1.-t*t) * pp* pp )
        w[npts-i] = w[i-1]
    if ( job == 0) :
        for i in range (0 , npts ) :
            x [ i ] = x [ i ] * ( b-a ) /2. + (b + a ) /2.
            w[ i ] = w[ i ] * ( b-a ) /2.
    if ( job == 1) :
        for i in range (0 , npts ) :
            xi = x [ i ]
            x[i ] = a *b * ( 1.+ xi )/ (b+a-( b-a ) * xi )
            w[ i ] = w[ i ] * 2.*a*b*b /( (b + a - ( b-a ) * xi ) *( b+a-(b-a)*xi))
    if ( job == 2) :
        for i in range (0 , npts ) :
            xi = x [ i ]
            x[i] = (b*xi+b+a+a) / (1.-xi)
            w[i] = w[i] *2.*( a + b)/((1.-xi)*(1.-xi))
def gaussint (func, no , min1, max1) :
    quadra = 0.
    gauss (no ,0. , min1, max1, x, w) # Returns pts (x) & wts (w)
    for n in range (0,no) :
        quadra += func(x[n]) * w[n] # Calculate integral
    return (quadra)


for i in range(3,max_in+1,2):
    result = gaussint(f1, i,vmin,vmax)
    print(" i ", i , " err ", abs(result-1 + 1/ME))
print("Enter and return any character to quit") 

print("the result of the integral is %f" %gaussint(f2, 10,0,1.))

error = (gaussint(f2, 2,vmin,vmax) - (1.-1./ME))/((1.-1./ME))

print "actual error of 2 %f" %error
print (1.-1./ME)

error = (gaussint(f2, 10,vmin,vmax) - (1.-1./ME))/((1.-1./ME))

print "actual error of 10 %f" %error





x1 = np.arange(2,50)
y1 = []
for i in xrange(len(x1)):
    y1.append(log10((gaussint(f2, x1[i],vmin,vmax)-(1.-1./ME)))-log10(1.-1./ME))
    print (log10((gaussint(f2, x1[i],vmin,vmax)-(1.-1./ME)))-log10(1.-1./ME))
    #y1.append(gaussint(f2, x1[i],vmin,vmax))
fig1 = plt.figure(1)

plt.plot(x1,y1)
plt.xlabel("number of terms")
plt.ylabel("Integral Value")
plt.title("plot of Gauss integration error for e^-x")
fig1.show()





