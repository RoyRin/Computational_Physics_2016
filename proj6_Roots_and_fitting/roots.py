# -*- coding: utf-8 -*-
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.optimize


def f1(x):
    return (sqrt(10.-x)*np.tan(np.sqrt(10.-x))-sqrt(x))
    
def f(x):# Function = 0?
    return 2.* np.cos(x)-x
    
def bisection( xminus , xplus , Nmax, eps, func) : # x+, x−, Nmax, error
    # for all intents and purposes this is doing a binary search tree - assuming the function does not intersect 0 more than once
    for it in range (0 , Nmax):
        x = (xplus + xminus) / 2. # Mid point
        print ("it" , it , " x " , x, " f(x) " , func(x))
        if ( func(xplus)* func(x) > 0. ) : # Root in other half
            xplus = x # Change x+ to x
        else :
            xminus = x # Change x− to x
        if ( abs ( func(x) ) < eps ) : # Converged?
            print ( "\n Root found with precision eps = " , eps)
            break
        if it == Nmax-1: # could not find a root
            print ("\n Root NOT found after Nmax iterations\n" )
    return x

def newtonRaphson(xo,imax, epsil, func ):
    x= xo
    dx = 3.0e-1
    
    for it in range (0 , imax + 1) :
        F = func(x)
        if ( abs (F) <= epsil ) : # Check for convergence
            print ( "\n Root found , F =" , F, " , tolerance eps = " , eps)
            break
        print ("Iteration # = ",it," x = ",x," f(x) =",F)
        df = (func(x+dx/2)-func(x-dx/2))/dx # Central diff
        dx = -F / df
        x += dx
    return x

xo = 0.004
eps = 1e-5 # Precision of zero
a = 0.0 ; b = 2* np.pi # Root in [ a , b]
imax = 100 # Max no . iterat ions

rootnewton = newtonRaphson(xo, imax, eps, f1)
print ("Root by newton Raphson method =" , rootnewton )
rootbisection = bisection(a,b,imax,eps,f1)
print ("Root by bisection=" , rootbisection )

rootscipy = sp.optimize.root(f1, xo, method = 'lm')
print ("Root by root.optimize.scipy=" , rootscipy.x[0] )

fig1 = plt.figure(1)
x1 = np.arange(a,b, 0.01)
plt.plot(x1,f1(x1))
fig1.show()