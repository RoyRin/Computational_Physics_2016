
import numpy
import pylab as p
from numpy import *
import numpy as np
import matplotlib.pyplot as plt 
from operator import add

#import matplotlib.animation as animation

#function, initial, time to iterate through, k,p are arguments for the function
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
    return x # x[0] is 0th derivative, x[1] is first derivative



def vibratingPendulumLagrange( x,t ,f,thetaDot) : # Force function - this is the derivative
    #y = zeros((2), float)
    fReturn = zeros((2),float)
    #y [ 0 ] = 3.;y [ 1 ] = -5. # intial derivative
    fReturn[0] = x[1] # y= x'
    #fReturn[1] = np.sin(x[0]*360./(2.*np.pi))/L # y' = -kx
    fReturn[1] = - (alpha * fReturn[0] )- ((w0**2 +(f*np.cos(w*t)))*np.sin(x[0])) # y' = -kx
    return fReturn
    
    



alpha = 0.1
w0 = 1.
w = 2.
thetaDot = []
thetaDot1 = []

numRepeats = 1 # the number of times that you want to plot the value for a particular f
# also used to be the waiting period of number of steps in time required to stabilize
steps = 50*(w0 *(2*np.pi)+0.1) + 0.
t = np.arange(0,steps)
theta_0 = np.array([1., 1.])

numOfF = 10000
df = 2.25/(numOfF*1.0)
flength = numRepeats*numOfF
fi = np.arange(0.,2.25,df)

f = np.arange(0,flength,1.) # need to make an array of 0.001, 0.0001, ...x150,  0.0002, till 2.25

for i in xrange(0,numOfF):
    for j in xrange(0,numRepeats): # repeat the same f, 150 times
        f[(i*numRepeats + j)] = fi[i] *1.0
ave = 0.0
for i in xrange(0, numOfF):
    vibratingPendulum = rk4_v1(vibratingPendulumLagrange,theta_0,t, fi[i],1.) # last two vars: f, ___
    '''
    for j in xrange(1, numRepeats+1):
        ave += abs(vibratingPendulum[-1*j][1])
        thetaDot.append(abs(vibratingPendulum[-1*j][1])) # add the last 150 time steps
    '''
    thetaDot.append(abs(vibratingPendulum[-1][1]))
    #thetaDot1.append(ave/(numRepeats*1.))
print len(f)
print len(thetaDot)

fig1 = plt.figure(1)
fig1.add_subplot( 1,1, 1 )
#plt.plot(f,thetaDot, 'ro' , label = "Phase" )
plt.scatter(f,thetaDot,s=0.9, label = "Phase" )

#plt.plot(f, 'ro' , label = "Phase" )
#plt.plot(t, 3*np.sin(t), 'bs', label = "actual solution")
#plt.plot(  t, xder, 'bs', label = "derivative" )
plt.xlabel("f" )
plt.ylabel("| d (theta)/dt | " )
plt.title( "Bifurcation Diagram of Dampened Pendulum" )
#plt.legend(bbox_to_anchor=(0.99, .99), loc='upper right', borderaxespad=0.)
plt.show()
'''
print len(fi)
print len(thetaDot1)
fig2 = plt.figure(2)
fig2.add_subplot( 1,1, 1 )
plt.plot(fi,thetaDot1, 'ro' , label = "Phase" )
#plt.plot(f, 'ro' , label = "Phase" )
#plt.plot(t, 3*np.sin(t), 'bs', label = "actual solution")
#plt.plot(  t, xder, 'bs', label = "derivative" )
plt.xlabel("f" )
plt.ylabel("| d \theta/dt | " )
plt.title( "Bifurcation Diagram of Dampened Pendulum - averaged " )
#plt.legend(bbox_to_anchor=(0.99, .99), loc='upper right', borderaxespad=0.)
plt.show()

'''
