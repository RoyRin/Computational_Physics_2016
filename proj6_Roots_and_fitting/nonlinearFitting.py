# -*- coding: utf-8 -*-
# Fit .py Linear least−squares f i t ; e . g . of matrix computation arrays
import pylab as p
from numpy import *
import numpy as np
from numpy.linalg import inv
from numpy.linalg import solve
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.optimize import curve_fit




#energy
Ei= np.array([0,25,50 ,75 ,100, 125, 150, 175, 200]) # energy
#experimental value
gEi = np.array([ 10.6, 16.0, 45.0, 83.5, 52.8, 19.9, 10.8, 8.25, 4.7])
#error
Error = np.array([9.34 ,17.9, 41.5, 85.5, 51.5, 21.5 ,10.8, 6.29, 4.14])
tab71 = np.zeros(shape=(9,4))
for i in xrange(1,10):
    tab71[i-1] = (i, Ei[i-1],gEi[i-1],Error[i-1])
    
print tab71

print tab71[1]



# we are trying to fit the data to:  f(e) = fr/((e-Er)^2 + G^2/4)
# we will put this in the form of f(x) = a1 / ((x-a2)^2 + a3 )
def func(x,a1,a2,a3):
    return a1/((x-a2)**2 + a3)


popt, pcov = curve_fit(func, Ei, gEi)
perr = np.sqrt(np.diag(pcov))

print "popt"
print popt
print "pcov"
print pcov

print perr

fig1 = plt.figure(1)
plt.title("data versus scipy.optimize curve fit")
plt.xlabel("Energy with some units")
plt.ylabel("Value" )

plt.plot(Ei, gEi, label = "Experimental Data")
x1 = np.arange(0, 200, 0.5)
y1 = func(x1,popt[0], popt[1], popt[2])
plt.plot(x1,y1, label = "Scipy Curve Fit")
plt.legend(bbox_to_anchor=(0.65, .85), loc=2, borderaxespad=0.)

fig1.show()


def dgda1(x,a1,a2,a3):
    return 1/((x-a2)**2 + a3)
def dgda2(x,a1,a2,a3):
    return 2*a1*(x-a2)/((x-a2)**2 + a3)**2
def dgda3(x,a1,a2,a3):
    return -a1/((x-a2)**2 + a3)**2
    
