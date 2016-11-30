

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

#equation to solve : Theta '' = -sin Theta / L

L = 1.
def singlePendulumLagrange( x,t ,k,b) : # Force function - this is the derivative
    #y = zeros((2), float)
    fReturn = zeros((2),float)
    #y [ 0 ] = 3.;y [ 1 ] = -5. # intial derivative
    fReturn[0] = x[1] # y= x'
    #fReturn[1] = np.sin(x[0]*360./(2.*np.pi))/L # y' = -kx
    fReturn[1] = -np.sin(x[0])/L # y' = -kx
    return fReturn

m1 = 2.
m2 = 1.
L1 = 2.
L2 = 1.
g = 9.81
def doublePendulumLagrange( x,t ,k,b) : # the derivative of this is taken, and 1 step at a time
    # solving for 
    #y = zeros((2), float)
    fReturn = zeros((4),float) # theta1, theta2, theta1', theta2'
    fReturn[0] = x[2] # z1' = theta1'
    fReturn[1] = x[3] # z2 ' = theta2 '
    
    #z3' = all of this>>
    fReturn[2] = ((-m2*L1*x[3]**2 *np.sin(x[0]-x[1])*np.cos(x[0]-x[1]))+ \
    g*m2*np.sin(x[1])*np.cos(x[0]-x[1]) - m2*L2*x[3]**2*np.sin(x[0]-x[1])- \
    (m1+m2)*g*np.sin(x[0]))/(L1*(m1+m2)-m2*L1*np.cos(x[0]-x[1])**2)   
    #fReturn[2] = fReturn[2]%(2*np.pi)
    #y [ 0 ] = 3.;y [ 1 ] = -5. # intial derivative
    #z4' = all of this>>
    fReturn[3] = ((m2*L2*x[3]**2 *np.sin(x[0]-x[1])*np.cos(x[0]-x[1]))+ \
    g*(m2+m1)*np.sin(x[0])*np.cos(x[0]-x[1]) + L1*x[3]**2*(m1+m2) *np.sin(x[0]-x[1])\
    - g*np.sin(x[1])*(m1+m2) ) / (L2*(m1+m2)-(m2*L1*np.cos(x[0]-x[1])**2) ) 
    #fReturn[1] = np.sin(x[0]*360./(2.*np.pi))/L # y' = -kx
    #fReturn[1] = -np.sin(x[0])/L # y' = -kx
    return fReturn




time = 6*np.pi
steps = 10000
t = np.arange(0,time, time/steps)
x_0 = np.array([np.pi/5., 0])



singlePendulum = rk4_v1(singlePendulumLagrange,x_0,t, 1.,1.)
singlePos = [(i[0]) for i in singlePendulum]
singleVel =  [(i[1]) for i in singlePendulum]






x_0_double = np.array([np.pi/4., 0,0.,0.]) # theta1, theta2, theta1', theta2' 
doublePendulum = rk4_v1(doublePendulumLagrange,x_0_double,t, 1.,1.)
doublePos1 = [(i[0]) for i in doublePendulum]
doublePos2 = [(i[1]) for i in doublePendulum]
doubleVel1 =  [(i[2]) for i in doublePendulum]
doubleVel2 =  [(i[3]) for i in doublePendulum]




'''
line, = ax.plot([],[], 'o-', lw=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.05,0.9,'',transform = ax.transAxes)
def init():
    line.set_data([],[])
    return line,
def animate(i):
    thisx = [0,doublePos1[i][0], doublePos2[i][0]]
    thisy = [0,doublePos1[i][1], doublePos2[i][1]]
    line.set_data(thisx,thisy)
    time_text.set_text(time_template % (i*dt))
    return line, time_text

ani = animation.FuncAnimation(fig,animate, np.arange(1, len(y)), interval =25, blit = True, init_func=init)
      
plt.show()
'''
    
#    figure( 2 )
fig2 = plt.figure(2)
fig2.add_subplot( 2,1, 1 )
plt.plot(  t, singlePos, 'r-o' , label = "position" )
#plt.plot(t, 3*np.sin(t), 'bs', label = "actual solution")
#plt.plot(  t, xder, 'bs', label = "derivative" )
plt.xlabel("time" )
plt.ylabel("position" )
plt.title( "Pendulum Position" )
plt.legend(bbox_to_anchor=(0.99, .99), loc='upper right', borderaxespad=0.)


fig2.add_subplot( 2,1, 2 )
plt.plot(  t, singleVel, 'r-o' , label = "velocity" )
#plt.plot(t, 3*np.sin(t), 'bs', label = "actual solution")
#plt.plot(  t, xder, 'bs', label = "derivative" )
plt.xlabel("time" )
plt.ylabel("velocity" )
plt.title( "Pendulum Position" )
plt.legend(bbox_to_anchor=(0.99, .99), loc='upper right', borderaxespad=0.)
plt.show()



fig3 = plt.figure(3)
fig3.add_subplot( 2,1, 1 )
plt.plot(  singlePos, singleVel, 'r-o' , label = "Phase" )
#plt.plot(t, 3*np.sin(t), 'bs', label = "actual solution")
#plt.plot(  t, xder, 'bs', label = "derivative" )
plt.xlabel("Position" )
plt.ylabel("velocity" )
plt.title( "Phase Space of a Single Pendulum" )
#plt.legend(bbox_to_anchor=(0.99, .99), loc='upper right', borderaxespad=0.)
plt.show()


#    figure( 2 )
fig4 = plt.figure(4)
fig4.add_subplot( 2,1, 1 )
plt.plot(  t, doublePos1, 'r-o' , label = "theta1" )
plt.plot(  t, doublePos2, 'r-o' , label = "theta2" )
#plt.plot(t, 3*np.sin(t), 'bs', label = "actual solution")
#plt.plot(  t, xder, 'bs', label = "derivative" )
plt.xlabel("time" )
plt.ylabel("Angle" )
plt.title( "Pendulum Position" )
plt.legend(bbox_to_anchor=(0.99, .99), loc='upper right', borderaxespad=0.)


fig4.add_subplot( 2,1, 2 )
plt.plot(  t, doubleVel1, 'r-o' , label = "Velocity of Pendulum1" )
plt.plot(  t, doubleVel2, 'bo' , label = "Velocity of Pendulum2" )
#plt.plot(t, 3*np.sin(t), 'bs', label = "actual solution")
#plt.plot(  t, xder, 'bs', label = "derivative" )
plt.xlabel("time" )
plt.ylabel("Angular Velocity" )
plt.title( "Pendulum Velocity" )
plt.legend(bbox_to_anchor=(0.99, .99), loc='upper right', borderaxespad=0.)
plt.show()


fig5 = plt.figure(5)
fig5.add_subplot( 1,1, 1 )
plt.plot(doublePos2,doubleVel2, 'r-o' , label = "Phase" )
#plt.plot(t, 3*np.sin(t), 'bs', label = "actual solution")
#plt.plot(  t, xder, 'bs', label = "derivative" )
plt.xlabel("Position" )
plt.ylabel("velocity" )
plt.title( "Phase Space of a double Pendulum - pendulum2" )
#plt.legend(bbox_to_anchor=(0.99, .99), loc='upper right', borderaxespad=0.)
plt.show()


fig6 = plt.figure(6)
fig6.add_subplot( 1,1, 1 )
plt.plot(doublePos1,doubleVel1, 'r-o' , label = "Phase" )
#plt.plot(t, 3*np.sin(t), 'bs', label = "actual solution")
#plt.plot(  t, xder, 'bs', label = "derivative" )
plt.xlabel("Position" )
plt.ylabel("velocity" )
plt.title( "Phase Space of a double Pendulum - mass 1" )
#plt.legend(bbox_to_anchor=(0.99, .99), loc='upper right', borderaxespad=0.)
plt.show()


fig, ax = plt.subplots()

x = [1, 2, 3, 4]
y = [5, 6, 7, 8]
print doublePos1[1]    
origin = (0,0)
mass1 = (L1*np.sin(doublePos1[0]),  -L1*np.cos(doublePos1[0]) )
mass2 = ( mass1[0] + L2*np.sin(doublePos2[0])  ,mass1[1] - L2*np.cos(doublePos2[0]))
line1, = ax.plot([origin[0], mass1[0]], [origin[1], mass1[1]] ,'r-' )
line2, = ax.plot([ mass1[0], mass2[0]], [mass1[1], mass2[1]] ,'r-' )
ax.set_xlim(-L2-L1, L1+L2) 
ax.set_ylim(-L2-L1, L1+L2) 
for t in range(0,len(t),100):#len(t)):

    mass1 = (L1*np.sin(doublePos1[t]),  -L1*np.cos(doublePos1[t]) )
    mass2 = ( mass1[0] + L2*np.sin(doublePos2[t])  ,mass1[1] - L2*np.cos(doublePos2[t]))
    #line = ax.plot([origin[0], mass1[0], mass2[0]], [origin[1], mass1[1], mass2[1]] )
   # line1.set_xdata([origin[0], mass1[0]] )
   # line1.set_ydata([origin[1], mass1[1]]  )
    line1.set_data([origin[0], mass1[0]] , [origin[1], mass1[1]]  )
    line2.set_data([mass1[0], mass2[0]] , [ mass1[1], mass2[1]]  )
   
    #points, = ax.plot(x, y, marker='o', linestyle='ls')

    print doublePos1[t]
    
    #line.clear()
 #   new_x = np.sin(doublePos1[t]) + L2*np.sin(doublePos2[t])
 #   new_y = -L1*np.cos(doublePos1[t])- L2*np.cos(doublePos2[t])
    #points.set_data(new_x, new_y)
    #new2_x = new_x+ L2*np.sin(doublePos2[t])
    #new2_y = new_y -L2*np.cos(doublePos2[t])
    #points.set_data(new2_x, new2_y)
    plt.pause(0.001)
plt.show()
