# -*- coding: utf-8 -*-
# rk4 . py 4th order Runge Kutta

# Initialization

import numpy
import pylab as p
from numpy import *
import numpy as np
import matplotlib.pyplot as plt 

a = 0. # intitial time value
b = 10. # max value in time
n = 100 # time steps
ydumb = zeros((2),float) 
y = zeros((2), float)
fReturn = zeros((2),float)
'''
k1 = zeros((2),float)
k2 = zeros ( ( 2 ) , float) 
k3 = zeros ((2), float)
k4 = zeros ( ( 2 ) , float)
'''

y [ 0 ] = 3. ; y [ 1 ] = -5.

t = a; h = (b-a ) /n ;

def f (t, y ) : # Force function - this is the derivative
    fReturn [0] = y[1]
    fReturn [1] = 1 #-1.*t # -100.* y [0] -2. * y [ 1 ] + 10. * sin ( 3. * t )
    return fReturn
    
def rk4 ( t , h , n) : #current time, step size, n dimension ; function and function derivative are stored globally 
    k1 = [0]*(n)
    k2 = [0]*(n)
    k3 = [0]*(n)
    k4 = [0]*(n)
    fR = [0]*(n)
    ydumb = [ 0 ] * ( n)
    fR = f ( t , y ) # Returns RHS’s
    for i in xrange (0 , n) :
        k1 [ i ] = h* fR [ i ]
    for i in xrange (0 , n) :
        ydumb[ i ] = y [ i ] + k1[ i ] /2.
        k2 = h* f ( t+h/2. , ydumb)
    for i in xrange (0 , n) :
        ydumb[ i ] = y [ i ] + k2[ i ] /2.
        k3 = h* f ( t+h/2. , ydumb)
    for i in xrange (0 , n) :
        ydumb[ i ] = y [ i ] + k3[ i ]
        k4 = h* f ( t+h, ydumb)
    for i in range (0 , 2) :
        y [ i ] = y [ i ] + (k1[ i ] + 2. * ( k2 [ i ] + k3 [ i ] ) + k4 [ i ] ) /6.
    return y
    
def f( t, y, fReturn ):         # function returns RHS, change
    fReturn[0] = y[1]           # 1st deriv of velocity is position
    fReturn[1] = -k*pow(y[0],p) # spring affects the change in velocity
    
def rKN(x, fx, n, hs):
    k1 = []
    k2 = []
    k3 = []
    k4 = []
    xk = []
    for i in range(n):
        k1.append(fx[i](x)*hs)
    for i in range(n):
        xk.append(x[i] + k1[i]*0.5)
    for i in range(n):
        k2.append(fx[i](xk)*hs)
    for i in range(n):
        xk[i] = x[i] + k2[i]*0.5
    for i in range(n):
        k3.append(fx[i](xk)*hs)
    for i in range(n):
        xk[i] = x[i] + k3[i]
    for i in range(n):
        k4.append(fx[i](xk)*hs)
    for i in range(n):
        x[i] = x[i] + (k1[i] + 2*(k2[i] + k3[i]) + k4[i])/6
    return x
#print rKN(1, f,2,0.1)
#x = -numpy.exp( 1.0 - numpy.cos( t ) )
'''    
while ( t < b) : # Time loop
    if ( ( t + h) > b):
        h = b - t # Last step
        y = rk4(t ,h,2)
        t = t + h
        rate (30)
        funct1.plot ( pos = ( t , y [ 0 ] ) )
        funct2.plot ( pos = ( t , y [ 1 ] ) )
        '''