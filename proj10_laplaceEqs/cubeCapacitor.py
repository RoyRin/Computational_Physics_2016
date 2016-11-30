""" From "COMPUTATIONAL PHYSICS", 3rd Ed, Enlarged Python eTextBook  
    by RH Landau, MJ Paez, and CC Bordeianu
    Copyright Wiley-VCH Verlag GmbH & Co. KGaA, Berlin;  Copyright R Landau,
    Oregon State Unv, MJ Paez, Univ Antioquia, C Bordeianu, Univ Bucharest, 2015.
    Support by National Science Foundation"""
   
# LaplaceLine.py:  Solve Laplace's eqtn, 3D matplot, close shell to quit

import matplotlib.pylab as p;
import mpl_toolkits.mplot3d as mplot3d
from mpl_toolkits.mplot3d import Axes3D
from numpy import *
import numpy as np
import matplotlib.pyplot as plt 

print("Initializing")
Nmax = 60; Niter = 30; V = zeros((Nmax, Nmax,Nmax), float)            # float maybe Float
capac = zeros((Nmax, Nmax,Nmax), float) # this is a mask that tells you where the capacitors are
#E = zeros((Nmax, Nmax, Nmax,4),float);
E = np.zeros((Nmax, Nmax, Nmax,4),float)
#for i in xrange( np.floor(((Nmax-1)/3), np.floor((Nmax-1)*2/3))):
print E[0][0][1][2]
''''
for i in xrange( Nmax/4, 3*Nmax/4):
    for j in xrange(Nmax/2 - Nmax/10 , Nmax/2 + Nmax/10):
        V[i][j] = 100
'''
pot = -100.
#Outerbox box
for i in xrange (0, Nmax):
    for j in xrange(0, Nmax):
        capac[i][j][0] = 1
        capac[i][j][Nmax-1] = 1
        capac[i][0][j] = 1
        capac[i][Nmax-1][j] = 1
        capac[0][i][j] = 1
        capac[Nmax-1][i][j] = 1
        V[i][j][0] = 0
        V[i][j][Nmax-1] = 0
        V[i][0][j] = 0
        V[i][Nmax-1][j] = 0
        V[0][i][j] = 0
        V[Nmax-1][i][j] = 0
        
mid = Nmax/2
side = Nmax/4
 #inner box box, edges = -100
'''
for i in xrange (mid - side, mid+side):
    for j in xrange(mid - side, mid + side):
        for k in xrange(mid- side, mid + side):  
            if ((i== mid -side) or (i== mid +side) or (j== mid -side) or (j== mid +side) or (k== mid -side)or (k== mid +side) ):
                capac[i][j][k] = 2
                V[i][j][k] = pot
            else: 
                capac[i][j][k] = 1
                V[i][j][k] = 0.
'''         

for i in xrange (mid - side, mid+side):
    for j in xrange(mid - side, mid + side):
        capac[i][j][mid -side] = 2
        capac[i][j][mid+side] = 2
        capac[i][mid -side][j] = 2
        capac[i][mid+side][j] = 2
        capac[mid-side][i][j] = 2
        capac[mid+side][i][j] = 2
        V[i][j][mid-side] = pot
        V[i][j][mid+side] = pot
        V[i][mid-side][j] = pot
        V[i][mid+side][j] = pot
        V[mid-side][i][j] = pot
        V[mid+side][i][j] = pot   

#inside the cube - equipotential charges
for i in xrange (mid - side, mid+side):
    for j in xrange(mid - side, mid + side):
        for k in xrange(mid-side, mid+side):
            if capac[i][j][k] != 2 : # inside of hte inner cube = equi potential
                capac[i][j][k] = 1
                V[i][j][k] = 0


#this is the line at y= 0;for the box 0,0 to 100,100
# for k in range(0, Nmax-1):  V[k,0] = 100.0 *sin(k*pi/100.)             # line at 100V
    
print " i got here at least"
for iter in range(Niter):                                 # iterations over algorithm
    if iter%10 == 0: print iter
    for i in range(1, Nmax-1):                                                
        for j in range(1,Nmax-1): 
            for k in range(1,Nmax -2):
                if capac[i][j][k] ==0:
                    V[i,j,k] = (V[i+1,j,k]+V[i-1,j,k]+V[i,j+1,k]+V[i,j-1,k] +V[i,j,k+1]+V[i,j,k-1]) /6 # relaxation


step = 1
print " i got here at least"
for i in range(1, Nmax-2):                                                
    for j in range(1,Nmax-2): 
        for k in range(1,Nmax -2):
            # x derivative - Ex
            if( ((i<mid +side)and (i>mid-side)) and ((j<mid +side)and (j>mid-side)) and ((k<mid +side)and (k>mid-side)) ):
                E[i,j,k,0] = 0
                E[i,j,k,1] = 0
                E[i,j,k,2] = 0
                E[i,j,k,3] = 0
            else:
                E[i,j,k,0] = -1*((V[i-1,j,k] - V[i,j,k]) + (V[i,j,k] - V[i+1,j,k]) )/(2* step)
            #y derivative - Ey
                E[i,j,k,1] = -1*((V[i,j-1,k] - V[i,j,k]) + (V[i,j,k] - V[i,j+1,k]) )/(2* step)
                #z derivative - Ez
                E[i,j,k,2] = -1*((V[i,j,k-1] - V[i,j,k]) + (V[i,j,k] - V[i,j,k+1]) )/(2* step)
                
                E[i,j,k,3] = np.sqrt( E[i,j,k,0]**2 +  E[i,j,k,1] **2 +  E[i,j,k,2]**2 )

# electric field to potential"
# E = -del * V
# E = -dV/dx -dV/dy - dV /dz



x = range(0, Nmax-1, 2);  y = range(0, Nmax-1, 2) ; z = range(0, Nmax-1, 2)       # plot every other point                        
X, Y = p.meshgrid(x,y)
#X, Y, Z = p.meshgrid(x,y,z)                   

def functz(V):                                             # Function returns V(x, y)
    z = V[X,Y,mid]                        
    return z
def Emid(E):                                             # Function returns V(x, y)
    z = E[X,Y,mid,3]                        
    return z
def Emidx(E):                                             # Function returns V(x, y)
    z = E[X,Y,mid,0]                        
    return z
def Emidy(E):                                             # Function returns V(x, y)
    z = E[X,Y,mid,1]                        
    return z
def Emidz(E):                                             # Function returns V(x, y)
    z = E[X,Y,mid,2]                        
    return z
    
def Estrength(E):                                             # Function returns V(x, y)
    z = E[X,Y,Z,3]                        
    return z   
    
#Z = functz(V)

Z = Emid(E)                          
#Z = Estrength(E) 

if False:
    fig = p.figure()                                                      # Create figure
    ax = Axes3D(fig)                                                       # plot axes
    ax.plot_wireframe(X, Y, Z, color = 'r')   
    #ax.imshow(Z) # red wireframe
    ax.set_xlabel('X')                                                       # label axes
    ax.set_ylabel('Y')
    ax.set_zlabel('Potential')
    p.title("Two Capacitor Problem - Relaxation")
    p.show()                                           # display fig, close shell to quit

#plt.imshow(Z[3])
xE = E[X,Y,mid,0]
yE = E[X,Y,mid,1]

Q = plt.quiver(xE,yE)
plt.show()
#plt.quiver(Q,units = 'xy',scale = 1 )
'''
contourer =np.zeros((Nmax, Nmax,Nmax), float)
for i in xrange(0, Nmax):
    for j in xrange(0, Nmax):
        for k in xrange(0, Nmax):
            contourer[i][j][k] = E[i][j][k][3]
Axes3D.contour(contourer)
'''
plt.title('Electric Field Distribution')
#plt.colorbar()
plt.show()