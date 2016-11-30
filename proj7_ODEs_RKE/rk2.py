# -*- coding: utf-8 -*-

import numpy
import pylab as p
from numpy import *
import numpy as np
import matplotlib.pyplot as plt 


#y = zeros((2), float)
#fReturn = zeros((2),float) # value , and derivative (with time)
#time = 100
#t1 = np.arange(0,50,0.1)

def Fext(x,t,a,b): # external force
    return 0
def exponential(x,t,a,b):
    return 2*x
def func87(x,t,k,p):
    fReturn = zeros((1),float)
    fReturn[0] = Fext(x,t,a,b) + k*x**(p-1)
    return fReturn
    
def spring ( x,t ,k,b) : # Force function - this is the derivative
    fReturn = zeros((2),float)
    #y [ 0 ] = 3.;y [ 1 ] = -5. # intial derivative
    fReturn[0] = x[1] # y= x'
    fReturn[1] = x[0] * k # y' = -kx
    return fReturn
    
def f ( x,t ,a,b) : # Force function - this is the derivative
    fReturn = zeros((2),float)
    y [ 0 ] = 3.;y [ 1 ] = -5. # intial derivative
    fReturn [0] = x[1]
    fReturn [1] = -100.* y [0] -2. * y [ 1 ] + 10. * sin ( 3. * t ) # second derivative
    return fReturn # second derivative

def RK2(f, x0, t0, step): #where f is the time derivative of x, a function of x and t
	k1 = step * f(x0, t0,-1,1)
	k2 = step * f(x0 + (k1/2), t0 + (step/2),-1,1)
	xnew = x0 + k2
	return xnew
	
def rk2( f, x0, t ): # f is the derivative, such that x' = f(x,t)
                    #x(t[0]) = x0
                        # array of time locations to solve for
    n = len( t ) #
    x = numpy.array( [ x0 ] * n )
    for i in xrange( n - 1 ):
        h = t[i+1] - t[i] # set the step size to be a time step
        k1 = h * f( x[i], t[i],-1,1 )
        k2 = h * f( x[i] + k1, t[i+1],-1,1)
        x[i+1] = x[i] + ( k1 + k2 ) / 2.0
    return x
def rk4 ( t , h , n) : #current time, step size, n dimension ; function and function derivative are stored globally 
    k1 = [ 0 ] * ( n)
    k2 = [ 0 ] * ( n)
    k3 = [ 0 ] * ( n)
    k4 = [ 0 ] * ( n)
    fR = [ 0 ] * ( n)
    ydumb = [ 0 ] * ( n)
    fR = f ( t , y ,1,1) # Returns RHS’s
    for i in range (0 , n) :
        k1 [ i ] = h* fR [ i ]
    for i in range (0 , n) :
        ydumb[ i ] = y [ i ] + k1[ i ] /2.
        k2 = h* f ( t+h/2. , ydumb,1,1)
    for i in range (0 , n) :
        ydumb[ i ] = y [ i ] + k2[ i ] /2.
        k3 = h* f ( t+h/2. , ydumb,1,1)
    for i in range (0 , n) :
        ydumb[ i ] = y [ i ] + k3[ i ]
        k4 = h* f ( t+h, ydumb,1,1)
    for i in range (0 , 2) :
        y [ i ] = y [ i ] + (k1[ i ] + 2. * ( k2 [ i ] + k3 [ i ] ) + k4 [ i ] ) /6.
    return y
    
time = 100
t = np.arange(0,time, 0.1)
x0 = 1.0

x_0 = np.array([1., 3.0]) # initial for the 1st derivative, and the function
print x_0[1]

RK2(spring, x_0, 10., 0.3)


springEq = rk2(spring,x_0,t)
print springEq
x = [(i[0]) for i in springEq]
xder =  [(i[1]) for i in springEq]
print  x

x_exponential = rk2( exponential, x0, t )
#x_rk = rk4(spring,x0,t)
x = -numpy.exp( 1.0 - numpy.cos( t ) )

#    figure( 1 )
fig1 = plt.figure(1)
fig1.add_subplot( 1, 1, 1 )
plt.plot(t, log10(x_exponential), 'r-o' )
plt.xlabel( '$t$' )
plt.ylabel( 'Log (10) of $x$' )
plt.title( 'Solutions of $dx/dt = 2.0 * x $' )
plt.legend( (  '$O(h^2)$ Runge-Kutta' ), loc='lower right' )
fig1.show()

#    figure( 2 )
fig2 = plt.figure(2)

fig2.add_subplot( 2,1, 1 )
plt.plot(  t, x, 'r-o' , label = "position" )
#plt.plot(  t, xder, 'bs', label = "derivative" )
plt.xlabel("time" )
plt.ylabel("position" )
plt.title( "spring equation solver x'' = -k* x " )
plt.legend(bbox_to_anchor=(0.99, .99), loc='upper right', borderaxespad=0.)


fig2.add_subplot( 2, 1, 2 )
plt.plot(  t, xder , 'r-o', label = "velocity" )
plt.xlabel( "time" )
plt.ylabel( "velocity" )
plt.title( "plot of velocity of solution to dx/dt = x" )
#plt.legend( ( '$O(h^2)$ Runge-Kutta' ), loc='upper right' ) 
plt.legend(bbox_to_anchor=(0.99, .99), loc='upper right', borderaxespad=0.)
fig2.show()