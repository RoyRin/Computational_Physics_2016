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
    
def rk4_Balls( f, initVal,t, k ,nval ): # f is the derivative, such that x' = f(x,t)
                        #x(t[0]) = x0 (array)
                        # we are solving for x as a function of time
                        # array of time locations to solve for
    n = len( t ) #
    m = 1.
    val = numpy.array( [initVal ] * n )
    for i in xrange( n - 1 ):
        h = t[i+1] - t[i] # set the step size to be a time step
        
        k1 = h* f( val[i], t[i],k, nval)
        k2 = h* f ( val[i]+k1/2. ,t[i] + h/2.,k, nval)
        k3 = h* f ( val[i]+k2/2. , t[i]+h/2.,k,  nval )
        k4 = h* f ( val[i]+k3, t[i]+h, k, nval)
        val[ i+1 ] = val[i] + (k1+ 2. * ( k2 + k3 ) + k4 ) /6.
    return val 
    
def Vwell(x,a,vo):
    if (abs(x) <= a):
        return vo
    else:
        return 0.
    
g = 9.81

def x(Vox,t):
    fReturn = zeros((2),float)
    fReturn[0] = Vox*t
    fReturn[1] = Vox
def y(Voy,t):
    fReturn = zeros((2),float)
    fReturn[0] = Voy*t -0.5 * g*t**2
    fReturn[1] = Voy - g*t


def fFx(k,n, y):
    m = 1.
    xvel = y[1]
    totalVel = (y[1]**2. + y[3]**2.)**0.5
    return -k *m*(abs(y[1]))**(n) *  (xvel/totalVel)
def fFy(k,n,y):
    m=1.
    yvel = y[3]
    totalVel = (y[1]**2. + y[3]**2.)**0.5
    return (-k  *m* (abs(y[3]))**((n)) *  (yvel/totalVel))
def funcBallsFallingAir(f,t, k, n): # f is the value
    # f0 = x(t)
    #f[1] = dx/dt
    #f[2] = y(t)
    #f[3] = dy/dt\
    fReturn = zeros((4),float)
    if fReturn[0] <0:
        fReturn[0] = 0.
    fReturn[0] = f[1]    # x
    #if fReturn[1] < 0:
    #    fReturn[1] = 0.
    fReturn[1] = fFx(k,n, f) # x dot
   # if fReturn[2] <0:
    #    fReturn[2] = 0.
    fReturn[2] = f[3] # y 
    #if fReturn[3] < 0:
    #    fReturn[3] = 0.
    fReturn[3] = fFy(k,n, f) - g # y dot
   
    return fReturn

timesteps = 1000.
maxtime = 5.
t1 = np.arange(0,maxtime,maxtime/timesteps)
vo = 20. # m/s
angle = 32. *2. * np.pi / 360. # in degrees

nval = np.array([0.0, 1.,1.5,2.0])
k = 1.5
m = 1.


maxDist = []
def findMaxDist(xpos,ypos):
    for i in xrange (1, len(xpos)):
        if ypos[i] < 0.:
            return xpos[i-1]
    return -1. # find the max distance for each through
  

initValue = np.array([0,vo*np.cos(angle), 0,vo*np.sin(angle)])
# x, vx, y, vy
print initValue

ballsAir = rk4_Balls(funcBallsFallingAir, initValue,t1, k, nval[0])
for i in xrange (0, len(t1)):
    if ballsAir[i][0] < 0:
        ballsAir[i][0]  = -0.1
    if ballsAir[i][2] < 0:
        ballsAir[i][2]  = -0.1  
xpos = [(i[0]) for i in ballsAir]
ypos=  [(i[2]) for i in ballsAir]
xvel = [(i[1]) for i in ballsAir]
yvel=  [(i[3]) for i in ballsAir]
print xpos
print ypos
print xvel
print yvel
fig1 = plt.figure(1)
fig1.add_subplot( 1, 1, 1 )
#plt.plot(t1, ypos ,  'r-o', label = "y pos ball trajectory" )
#plt.plot(t1, xpos ,  'k', label = "x pos ball trajectory" )
plt.plot(xpos, ypos ,  'k', label = "ball trajectory" )
plt.xlabel( 'x position' )
plt.ylabel( 'y position' )
plt.title( "Ball being thrown with n = %f" %nval[0] )
plt.legend(bbox_to_anchor=(0.5, .25), loc='upper right', borderaxespad=0.)
fig1.show() 


print nval[3]
angles = np.arange(0,90.,.3)
for i in xrange( 0, len(angles)):
    angle = angles[i] *2. * np.pi / 360. # in degrees
    initValue = np.array([0,vo*np.cos(angle), 0,vo*np.sin(angle)])
    ballsAir = rk4_Balls(funcBallsFallingAir, initValue,t1, k, nval[1])
    xpos = [(s[0]) for s in ballsAir]
    ypos=  [(s[2]) for s in ballsAir]
    maxDist.append(findMaxDist(xpos, ypos))
    if (angles[i] >89.):
        print "the terminal velocity for:"
        print angles[i]
        print (ballsAir[len(ballsAir)-1][1]**2 + ballsAir[len(ballsAir)-1][3]**2 ) **0.5 # print the terminal velocity for launching directly up


fig2 = plt.figure(2)
fig2.add_subplot( 1, 1, 1 )
#plt.plot(t1, ypos ,  'r-o', label = "y pos ball trajectory" )
#plt.plot(t1, xpos ,  'bs', label = "x pos ball trajectory" )
plt.plot(angles, maxDist ,  'gs', label = "ball trajectory" )
plt.xlabel( 'angle' )
plt.ylabel( 'max Distance' )
plt.title( "Plot of max distance versus angle for n = %f" %nval[1] )
plt.legend(bbox_to_anchor=(0.95, .95), loc='upper right', borderaxespad=0.)
fig2.show() 

