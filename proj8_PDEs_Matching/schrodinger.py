# -*- coding: utf-8 -*-
import numpy
import pylab as p
from numpy import *
import numpy as np
import matplotlib.pyplot as plt 
from operator import add
def rk2( f, x0, t ,k,p): # f is the derivative, such that x' = f(x,t)
                        #x(t[0]) = x0 (it is in array format)
                        # array of time locations to solve for
                        #
    n = len( t ) #
    x = numpy.array( [ x0 ] * n )
    for i in xrange( n - 1 ):
        h = t[i+1] - t[i] # set the step size to be a time step
        k1 = h * f( x[i], t[i],k,p )
        k2 = h * f( x[i] + k1, t[i+1],k,p)
        x[i+1] = x[i] + ( k1 + k2 ) / 2.0
    return x
    
def rk4_schrodinger( f, initVal,x, V, a, depth, E ): # f is the derivative, such that x' = f(x,t)
                        #x(t[0]) = x0 (array)
                        # we are solving for x as a function of time
                        # array of time locations to solve for
    n = len( x ) #
    m = 1.
    val = numpy.array( [initVal ] * n )
    for i in xrange( n - 1 ):
        h = x[i+1] - x[i] # set the step size to be a time step
        
        k1 = h* f( val[i], x[i],m,V(x[i],a,depth),E )
        k2 = h* f ( val[i]+k1/2. ,x[i] + h/2.,m, V(x[i],a,depth), E)
        k3 = h* f ( val[i]+k2/2. , x[i]+h/2,m,V(x[i],a,depth), E)
        k4 = h* f ( val[i]+k3, x[i]+h,m, V(x[i],a,depth), E)
        val[ i+1 ] = val[i] + (k1+ 2. * ( k2 + k3 ) + k4 ) /6.
    return val 
def Vwell(x,a,vo):
    if (abs(x) <= a):
        return vo
    else:
        return 0.
def schrodinger(chi,x,  m, Vval,E):
    k = abs(2 * m /h**2 * E)**.5
    # solves for one value of chi, V, etc.
    fReturn = zeros((2),float) # 2nd order ODE
    fReturn[0] = chi[1] # y = chi '
    fReturn[1] = (k**2 + ((2*m/h)*Vval))* chi[0] 
    return fReturn
    
def func825(x,t,k,p):
    m = 1.
    fReturn = zeros((2),float)
    # solves for one value of x,t, etc.
    fReturn[1] = Fext(x,t,k,p)/m + k*x[0]**(p-1) /m # 2nd derivative
    fReturn[0] = x[1] # first derivtive
    return fReturn
    
def bisection( xminus , xplus , Nmax, eps, data) : # x+, x−, Nmax, error
    # for all intents and purposes this is doing a binary search tree - assuming the function does not intersect 0 more than once
    for it in range (0 , Nmax):
        x = floor((xplus + xminus) / 2.) # Mid point
        print ("it" , it , " x " , x, " f(x) " , data[x] )
        if (data[xplus]* data[x] > 0. ) : # Root in other half
            xplus = x # Change x+ to x
        else :
            xminus = x # Change x− to x
        if ( abs (data[x] ) < eps ) : # Converged?
            print ( "\n Root found with precision eps = " , eps)
            break
        if it == Nmax-1: # could not find a root
            print ("\n Root NOT found after Nmax iterations\n" )
    return x
    

Xmax = 15.
steps = 1000
m = 1.
#h= 4.135 * 10**-15 # eV s #6.626 * 10**-34 J s
h  = 1. #* 10**-3
k =1.
a = 1. * np.pi# width of the well
vo = -10. #ev depth of well
#vo = -10. # depth of the well
matchX = a
E = vo * 1./2. # first guess 
#k = (2*m/h**2 * E)**0.5

LeftSchrod = np.exp(-Xmax)
RightSchrod = np.exp(-Xmax) # boundary conditions

# left side
#leftX= np.arange(-Xmax, -a , abs(Xmax -a) /(steps*1.))
leftX1= np.arange(-Xmax, -a , abs(Xmax -a  ) /(steps*1.))
middX = np.arange(-a, matchX, 2.* (a + matchX) /steps *1.)
leftX = np.append(leftX1, middX)

leftrkSchrodinger = rk4_schrodinger(schrodinger, np.array([LeftSchrod, 0]),leftX,Vwell,a,vo, E)
leftposSchro = [(i[0]) for i in leftrkSchrodinger]
leftvelSchro=  [(i[1]) for i in leftrkSchrodinger]

#right side
rightX = np.arange(-Xmax, -matchX , abs(Xmax-matchX) /(steps*1.))
rightX = -1. * rightX

rightrkSchrodinger = rk4_schrodinger(schrodinger, np.array([RightSchrod, 0]),rightX,Vwell,a,vo,  E)
rightposSchro = [(i[0]) for i in rightrkSchrodinger]
rightvelSchro=  [(i[1]) for i in rightrkSchrodinger]

def logDer(pos, vel):
    return (vel*1.)/(pos * 1.)
def error (leftpos, leftvel, rightpos, rightvel):
    return (( logDer(leftpos,leftvel) - logDer(rightpos, rightvel))/ ( logDer(leftpos,leftvel) + logDer(rightpos, rightvel)))

print error (leftposSchro[steps-1], leftvelSchro[steps-1], rightposSchro[steps-1], rightvelSchro[steps-1])


print "asadasdad hellow to the other side"
EArr = np.arange(vo , abs(vo) ,.06)
errorArr = []
for q in xrange(0, len(EArr)):
    leftrkSchrodinger1 = rk4_schrodinger(schrodinger, np.array([LeftSchrod, 0]),leftX,Vwell,a,vo, EArr[q])
    #right side
    rightrkSchrodinger1 = rk4_schrodinger(schrodinger, np.array([RightSchrod, 0]),rightX,Vwell,a,vo,  EArr[q])
    errorArr.append(error(leftrkSchrodinger1[-1][0], leftrkSchrodinger1[-1][1], rightrkSchrodinger1[-1][0], rightrkSchrodinger1[-1][1]))
print errorArr

epsilon = 10**-1

#print bisection(vo,abs(vo),200, epsilon, EArr)
#print "done"

Eo = abs(vo)* 1./2. #/2. # first E guess
maxsteps = 500.
count = 0
err= 10.

# essentially do euler's rule to step forward/back to find a root
while(abs(err) > epsilon):
    leftrkSchrodinger1 = rk4_schrodinger(schrodinger, np.array([LeftSchrod, 0]),leftX,Vwell,a,vo, Eo)
    #right side
    rightrkSchrodinger1 = rk4_schrodinger(schrodinger, np.array([RightSchrod, 0]),rightX,Vwell,a,vo,  Eo)
    err = (error(leftrkSchrodinger1[-1][0], leftrkSchrodinger1[-1][1], rightrkSchrodinger1[-1][0], rightrkSchrodinger1[-1][1]))
    print "error asdasdasd"
    print err
    print "energy trial"
    print Eo
    Eo = Eo - (err /abs(err)) *(2.*vo/maxsteps)
    count+= 1
    print ("counting %d" , count)
    if count >= 300:
        break
print err
print "the eigen value is : "
print Eo

print "count"
print count
print len(leftX)

fig1 = plt.figure(1)
fig1.add_subplot( 1, 1, 1 )
plt.plot(leftX, leftposSchro ,  'k', label = "Wave Function from left" )
plt.plot(rightX, rightposSchro ,  'k', label = "Wave Function from right" )
#plt.plot(midX, middleposSchro ,  'bs', label = "Wave Function" )
#plt.plot(x, newPos4p2 ,  'r-o', label = "stabilised position" )
plt.xlabel( 'time' )
plt.ylabel( 'Wave Function' )
plt.title( 'RK-4 solution for schrodinger for some arbitrary E' )
plt.legend(bbox_to_anchor=(0.99, .99), loc='upper right', borderaxespad=0.)
fig1.show() 



fig2 = plt.figure(2)
fig2.add_subplot( 1, 1, 1 )
plt.plot(EArr, errorArr ,  'r-o', label = "Error for different E's" )
plt.xlabel( 'Energy (eigen value)' )
plt.ylabel( 'Error' )
plt.title( 'Error function for different E values' )
plt.legend(bbox_to_anchor=(0.99, .99), loc='upper right', borderaxespad=0.)
fig2.show() 



print len(leftX) 
print len(leftrkSchrodinger1)
bestposleft = [(i[0]) for i in leftrkSchrodinger1]
bestposright = [(i[0]) for i in rightrkSchrodinger1]

fig3 = plt.figure(3)
fig3.add_subplot( 1, 1, 1 )
plt.plot(leftX, bestposleft ,  'k', label = "best E" )
plt.plot(rightX, bestposright ,  'k', label = "Best E" )

plt.xlabel( 'position' )
plt.ylabel( 'Wave' )
plt.title( 'Wave function for best E values' )
plt.legend(bbox_to_anchor=(0.99, .99), loc='upper right', borderaxespad=0.)
fig3.show() 
