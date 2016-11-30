""" From "COMPUTATIONAL PHYSICS", 3rd Ed, Enlarged Python eTextBook  
    by RH Landau, MJ Paez, and CC Bordeianu
    Copyright Wiley-VCH Verlag GmbH & Co. KGaA, Berlin;  Copyright R Landau,
    Oregon State Unv, MJ Paez, Univ Antioquia, C Bordeianu, Univ Bucharest, 2015.
    Support by National Science Foundation"""
   
# LaplaceLine.py:  Solve Laplace's eqtn, 3D matplot, close shell to quit

import matplotlib.pylab as p;
from mpl_toolkits.mplot3d import Axes3D
from numpy import *
import numpy as np

import matplotlib.pyplot as plt 

print("Initializing")
Nmax = 400; Niter = 70; V = zeros((Nmax, Nmax), float)            # float maybe Float
capac = zeros((Nmax, Nmax), float) # this is a mask that tells you where the capacitors are
charge = zeros((Nmax, Nmax), float)
#for i in xrange( np.floor(((Nmax-1)/3), np.floor((Nmax-1)*2/3))):

''''
for i in xrange( Nmax/4, 3*Nmax/4):
    for j in xrange(Nmax/2 - Nmax/10 , Nmax/2 + Nmax/10):
        V[i][j] = 100
'''
mid = Nmax/2
width = 5
offset  = 10
xwall = 4


# 1 plate
for j in xrange(xwall, Nmax-xwall):
    for i in xrange(mid - offset - width , mid - offset + width):
        V[i][j] = -100
        capac[i][j] =1 # if it is inside the conductor
        if ((j == xwall) or (j == Nmax - xwall -1) or ( i == mid - offset - width) or (i == mid - offset + width-1)):
            capac[i][j] = 2 # 2 if it is an edge
for j in xrange(xwall, Nmax -xwall ):
    for i in xrange(mid + offset - width , mid + offset + width):
        V[i][j] = 100
        capac[i][j] =1
        if ((j == xwall) or (j == Nmax-xwall-1 ) or ( i == mid + offset - width) or (i == mid + offset + width-1 )):
            capac[i][j] = 2

print V

#this is the line at y= 0;for the box 0,0 to 100,100
# for k in range(0, Nmax-1):  V[k,0] = 100.0 *sin(k*pi/100.)             # line at 100V
    
for iter in range(Niter):                                 # iterations over algorithm
    if iter%10 == 0: print iter
    for i in range(1, Nmax-2):                                                
        for j in range(1,Nmax-2): 
            if capac[i][j] ==0:
                V[i,j] = 0.25*(V[i+1,j]+V[i-1,j]+V[i,j+1]+V[i,j-1])  # relaxation

                
for iter in range(Niter):                                 # iterations over algorithm
    if iter%10 == 0: print iter
    for i in range(0, Nmax-1):                                                
        for j in range(0,Nmax-1): 
            if capac[i][j] ==2 :              
                charge[i,j] = -(0.25*(V[i+1,j]+V[i-1,j]+V[i,j+1]+V[i,j-1])- 4*V[i,j])/np.pi # relaxation


   
x = range(Nmax/3, 2*(Nmax )/3, 2);  y = range(0, Nmax/2, 2)                # plot every other point                        
X, Y = p.meshgrid(x,y)                 

def functz(V):                                             # Function returns V(x, y)
    z = V[X,Y]                        
    return z

'''
Z = functz(V)   
Char= functz(charge)
plt.imshow(charge)
plt.title('Charge Distribution')
plt.colorbar()
plt.show()
'''
Z = functz(V)   
Char= functz(charge)

plt.imshow(Z)
plt.title('Potential Distribution')
plt.colorbar()
plt.show()

'''
              
                  
fig1 = plt.figure(1)                                                      # Create figure
#ax = Axes3D(fig1)                                                       # plot axes
ax = fig1.gca(projection='3d')
ax.plot_wireframe(X, Y, Z, color = 'r')                               # red wireframe
ax.set_xlabel('X')                                                       # label axes
ax.set_ylabel('Y')
ax.set_zlabel('Potential')
plt.suptitle("Two Capacitor Problem - Relaxation")
fig1.show()                                           # display fig, close shell to quit


'''


fig2 = plt.figure(2)                                                      # Create figure
#ax = Axes3D(fig1)                                                       # plot axes
ax = fig2.gca(projection='3d')
ax.plot_wireframe(X, Y, Char, color = 'r')                               # red wireframe
ax.set_xlabel('X')                                                       # label axes
ax.set_ylabel('Y')
ax.set_zlabel('Charge')
plt.suptitle("Two Capacitor Problem - Relaxation of Charges")
fig2.show()        
#plt.suptitle("Two capacitor problem - relaxation")
#fig4.show()

