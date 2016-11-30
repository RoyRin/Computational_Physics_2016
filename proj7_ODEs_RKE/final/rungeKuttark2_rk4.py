# -*- coding: utf-8 -*-

import numpy
import pylab as p
from numpy import *
import numpy as np
import matplotlib.pyplot as plt 
from operator import add
def rk2( f, x0, t ,k,p): # f is the derivative, such that x' = f(x,t)
                    #x(t[0]) = x0
                        # array of time locations to solve for
    n = len( t ) #
    x = numpy.array( [ x0 ] * n )
    for i in xrange( n - 1 ):
        h = t[i+1] - t[i] # set the step size to be a time step
        k1 = h * f( x[i], t[i],k,p )
        k2 = h * f( x[i] + k1, t[i+1],k,p)
        x[i+1] = x[i] + ( k1 + k2 ) / 2.0
    return x
    
def rk4_v1( f, x0, t,k,p ): # f is the derivative, such that x' = f(x,t)
                    #x(t[0]) = x0
                    # we are solving for x as a function of time
                        # array of time locations to solve for
    n = len( t ) #
    x = numpy.array( [ x0 ] * n )
    for i in xrange( n - 1 ):
        h = t[i+1] - t[i] # set the step size to be a time step
        
        k1 = h* f( x[i], t[i],k,p )
        k2 = h* f ( x[i]+k1/2. ,t[i] + h/2.,k,p)
        k3 = h* f ( x[i]+k2/2. , t[i]+h/2,k,p)
        k4 = h* f ( x[i]+k3, t[i]+h,k,p)
        x[ i+1 ] = x[i] + (k1+ 2. * ( k2 + k3 ) + k4 ) /6.
    return x

def Fext(x,t,a,b): # external force
    return 0
def exponential(x,t,a,b):
    return 2*x
def func87(x,t,k,p):
    fReturn = zeros((1),float)
    fReturn[0] = Fext(x,t,k,p) + k*x**(p-1)
    return fReturn    
def func825(x,t,k,p):
    m = 1.
    fReturn = zeros((2),float) #    This should be the same number as there are derivatives (2)
    fReturn[1] = Fext(x,t,k,p)/m + k*x[0]**(p-1) /m  # 2nd derivative y' = x'' /m
    fReturn[0] = x[1] # first derivative :  y = x'
    return fReturn
    
def spring ( x,t ,k,b) : # Force function - this is the derivative
    #y = zeros((2), float)
    fReturn = zeros((2),float)
    #y [ 0 ] = 3.;y [ 1 ] = -5. # intial derivative
    fReturn[0] = x[1] # y= x'
    fReturn[1] = x[0] * k # y' = -kx
    return fReturn
    
time = 10*np.pi
steps = 1000
t = np.arange(0,time, time/steps)
x0 = 1.0

x_0 = np.array([5., 0.]) # initial for the 1st derivative, and the function

k = -1.

eq825 = rk2(func825,x_0,t,k,4)
pos = [(i[0]) for i in eq825]
vel =  [(i[1]) for i in eq825]
print vel

eq825_rk4 = rk4_v1(func825,x_0,t,k,2)
pos4p2 = [(i[0]) for i in eq825_rk4]
vel4p2 =  [(i[1]) for i in eq825_rk4]

eq825_rk4p6 = rk4_v1(func825,x_0,t,k,6)
pos4p6 = [(i[0]) for i in eq825_rk4p6]
vel4p6 =  [(i[1]) for i in eq825_rk4p6]


# 8.8 - 3 - show that the period of a function for p>2 is non-isochronous - and does depend on amplitude
def findPeriod(t, val):
    xo = []
    for i in xrange(0, len(val)-5 , 5):
        if((val[i] * val[i+5]) <=0.):
            xo.append(t[i] )
    per = 0.
    if len(xo) <=1:
        return 0.
    for i in xrange(0, len(xo)-1):
        per += xo[i+1] -xo[i]
    return per/(( len(xo))-1 )

print "period is : \n"
print (findPeriod(t, pos4p2))    
amp = np.arange(1,15.)
periods = []    
k= -2
for i in xrange (0, len(amp)):
    Period_rk4p6 = rk4_v1(func825,np.array([amp[i] *1., 0]),t,k,6)
    perPos4p6 = [(i[0]) for i in Period_rk4p6]
    periods.append( (findPeriod(t, perPos4p6)))


print "periods"
print periods
fig1 = plt.figure(1)
fig1.add_subplot( 1, 1, 1 )
plt.plot(amp, periods, 'r-o' )
plt.xlabel( 'intial amplitude' )
plt.ylabel( 'periods' )
plt.title( 'Periods verus Amplitude' )
plt.legend(bbox_to_anchor=(0.99, .99), loc='upper right', borderaxespad=0.)
fig1.show()      
        
        
        
x_exponential = rk2( exponential, x0, t,1,1 )
#x_rk = rk4(spring,x0,t)
'''
#    figure( 1 )
fig1 = plt.figure(1)
fig1.add_subplot( 1, 1, 1 )
plt.plot(t, log10(x_exponential), 'r-o' )
plt.xlabel( '$t$' )
plt.ylabel( 'Log (10) of $x$' )
plt.title( 'Solutions of $dx/dt = 2.0 * x $' )
plt.legend( (  '$O(h^2)$ Runge-Kutta' ), loc='lower right' )
fig1.show()
'''

# spring equation through RK-2
#    figure( 2 )
fig2 = plt.figure(2)
fig2.add_subplot( 2,1, 1 )
plt.plot(  t, pos4p2, 'r-o' , label = "position" )
#plt.plot(t, 3*np.sin(t), 'bs', label = "actual solution")
#plt.plot(  t, xder, 'bs', label = "derivative" )
plt.xlabel("time" )
plt.ylabel("position" )
plt.title( "Equation 8.27 position with  p =2" )
plt.legend(bbox_to_anchor=(0.99, .99), loc='upper right', borderaxespad=0.)


fig2.add_subplot( 2, 1, 2 )
plt.plot(  t, vel4p2 , 'r-o', label = "velocity" )
#plt.plot(t, 3* np.cos(t), 'bs', label = "actual solution")
plt.xlabel( "time" )
plt.ylabel( "velocity" )
plt.title( "Equation 8.27 velocity with  p =4"  )
#plt.legend( ( '$O(h^2)$ Runge-Kutta' ), loc='upper right' ) 
plt.legend(bbox_to_anchor=(0.99, .99), loc='upper right', borderaxespad=0.)
fig2.show()



#spring equation for rk-4
fig3 = plt.figure(3)

fig3.add_subplot( 2,1, 1 )
plt.plot(  t, pos4p2, 'r-o' , label = "position p = 2" )
plt.plot(  t, vel4p2 , 'r-o', label = "velocity for p =2" )
#plt.plot(  t, pos4p6, 'bs' , label = "position p =6" )
#plt.plot(t, 3*np.sin(t), 'bs', label = "actual solution")
#plt.plot(  t, xder, 'bs', label = "derivative" )
plt.xlabel("time" )
plt.ylabel("value" )
plt.title( "Equation 8.27 with  p =2 - rk4" )
plt.legend(bbox_to_anchor=(0.99, .99), loc='upper right', borderaxespad=0.)


fig3.add_subplot( 2, 1, 2 )
#plt.plot(  t, vel4p2 , 'r-o', label = "velocity for p =2" )
plt.plot(  t, pos4p6, 'bs' , label = "position p =6" )
plt.plot(  t, vel4p6 , 'bs', label = "velocity for p =6" )
#plt.plot(t, 3* np.cos(t), 'bs', label = "actual solution")
plt.xlabel( "time" )
plt.ylabel( "value" )
plt.title( "Equation 8.27 with  p =6 - rk4" )
#plt.legend( ( '$O(h^2)$ Runge-Kutta' ), loc='upper right' ) 
plt.legend(bbox_to_anchor=(0.99, .99), loc='upper right', borderaxespad=0.)
fig3.show()





# plot the energies of the function
#8.8.1 part
def V(x,k,p):
    Vret =x[:]
    for i in xrange(0, len(x)):
        Vret[i] = -k* x[i]**p / p
    return  Vret 
def KE( m,v):
    KEret =v[:]
    for i in xrange(0, len(v)):
        KEret[i] = (m * v[i]*v[i])/2.
    return  KEret 
    
   # return  [i*i*m/2.  for i in v] 
def TotalE(x,v,m, k,p):
    Energy = KE(m,v)[:]
    Ven = V(x,k,p)[:]
    for i in xrange(0, len (x)):
        Energy[i] += Ven[i]
    return Energy

E2 = TotalE(pos4p2,vel4p2,1.,-1.,2.)

print "\n size of total E %d" %len(E2) 
print "\n size of time %d" %len(t) 
fig4 = plt.figure(4)
fig4.add_subplot( 1, 1, 1 )
#plt.plot(t, pos4p2,  'g--', label = "Pos" )
#plt.plot(t, vel4p2,  'g--', label = "Vel" )

plt.plot(t, E2,  'bs', label = "Energy" )

#plt.plot(t, KE(1,vel4p2),  'r-o', label = "KE" )
#plt.plot(t, V(pos4p2,-1,2.),  'r-o', label = "PE" )
plt.xlabel( 'time' )
plt.ylabel( 'Energy' )
plt.title( 'Energy Plot' )
plt.legend(bbox_to_anchor=(0.99, .99), loc='upper right', borderaxespad=0.)
fig4.show() 



fig5 = plt.figure(5)
fig5.add_subplot( 1, 1, 1 )
plt.plot(t, -np.log10(abs(((E2)-E2[0])/E2[0]) + 1e-9) ,  'r-o', label = "Energy Stability" )
plt.xlabel( 'time' )
plt.ylabel( 'Log of Change in Energy' )
plt.title( 'Energy Stability Plot' )
plt.legend(bbox_to_anchor=(0.99, .99), loc='upper right', borderaxespad=0.)
fig5.show() 

# add in energy conservation, by scaling the energy by the energy at t = 0

energyScale = E2[:]
for i in xrange(0,len(energyScale)):
    energyScale[i] = energyScale[i]/E2[0]
newVel4p2 = vel4p2[:]
for i in xrange (0, len(vel4p2)):
    newVel4p2[i] = newVel4p2[i]/ energyScale[i]**0.5
    
newPos4p2 = pos4p2[:]
for i in xrange (0, len(pos4p6)):
    newPos4p2[i] = newPos4p2[i]/ energyScale[i]**0.5
    

    
fig6 = plt.figure(6)
fig6.add_subplot( 1, 1, 1 )
plt.plot(t, newVel4p2 ,  'r-o', label = "stabilised Velocity" )
plt.plot(t, newPos4p2 ,  'r-o', label = "stabilised position" )
plt.xlabel( 'time' )
plt.ylabel( 'Log of Change in Energy' )
plt.title( 'Normalised Energy Plot' )
plt.legend(bbox_to_anchor=(0.99, .99), loc='upper right', borderaxespad=0.)
fig6.show() 


ENorm = TotalE(newPos4p2,newVel4p2,1.,-1.,2.)

fig7 = plt.figure(7)
fig7.add_subplot( 1, 1, 1 )

plt.plot(t, -np.log10(abs(((ENorm)-ENorm[0])/ENorm[0]) + 1e-9) ,  'r-o', label = "Energy Stability" )

plt.xlabel( 'time' )
plt.ylabel( 'Log of Change in Energy' )
plt.title( 'Normalized Energy Stability Plot' )
plt.legend(bbox_to_anchor=(0.99, .99), loc='upper right', borderaxespad=0.)
fig7.show() 

# what is the average value of the energies
# use this to show the virial theorem <KE> = <PE> * p/2
def AveVal(x):
    summer =0.
    for i in xrange(0,len(x)):
        summer += x[i]
    return summer/len(x)
AveKE = AveVal(KE(1,vel4p6))
AvePE = AveVal(V(pos4p6,-1,6))
print "======================="
print AveKE
print AvePE *3