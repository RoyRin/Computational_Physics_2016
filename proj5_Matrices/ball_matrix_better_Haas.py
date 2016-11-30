""" From "COMPUTATIONAL PHYSICS", 3rd Ed, Enlarged Python eTextBook  
    by RH Landau, MJ Paez, and CC Bordeianu
    Copyright Wiley-VCH Verlag GmbH & Co. KGaA, Berlin;  Copyright R Landau,
    Oregon State Unv, MJ Paez, Univ Antioquia, C Bordeianu, Univ Bucharest, 2015.
    Support by National Science Foundation"""

# NewtonNDanimate.py  MultiDimension Newton Search
import numpy as np
import matplotlib.pyplot as plt
import sys

n = 9
deriv = np.zeros( (n, n), float)
f = np.zeros(n, float)

L=1.003     #total length
L1=10       #length of rods
L2=4
L3=7

if (L1+L2+L3 < L):
    print("That won't work too good...",str(L1+L2+L3)+"<"+str(L))
    sys.exit()
W1=4        #mass 1
W2=20       #mass 2

# we are solving for x for the equations on 118
x = np.array([-0.5, -0.5, -0.5, 0.5, 0.5, 0.5, W1+W2, W1+W2, W1+W2]) #starting guesses
# sin @1, sin@2, sin @3, cos @1, cos @2, cos @3, T1, T2, T3
def plotconfig():
    #for obj in scene.objects: obj.visible=0 # to erase the previous configuration
    xa = L1*x[3]                # L1*cos(th1)
    ya = -abs(L1*x[0])                # L1 sin(th1)
    xb = xa+L2*x[4]             # L1*cos(th1)+L2*cos(th2)
    yb = -abs(L1*x[0]+L2*x[1])             # L1*sin(th1)+L2*sin(th2)
    xc = xb+L3*x[5]             # L1*cos(th1)+L2*cos(th2)+L3*cos(th3)
    yc = L1*x[0]+L2*x[1]-L3*x[2]             # L1*sin(th1)+L2*sin(th2)-L3*sin(th3)
    plt.axes()
    plt.cla()
    circlea = plt.Circle((xa,ya), radius=0.1, fc='y')
    plt.gca().add_patch(circlea)
    circleb = plt.Circle((xb,yb), radius=0.1, fc='y')
    plt.gca().add_patch(circleb)
    circlec = plt.Circle((xc,yc), radius=0.1, fc='r')
    plt.gca().add_patch(circlec)
    circled = plt.Circle((0,0), radius=0.1, fc='r')
    plt.gca().add_patch(circled)
    linetop = plt.Line2D((0,xc), (0,yc), lw=.5, c='r')
    plt.gca().add_line(linetop)
    lineab = plt.Line2D((xa,xb), (ya,yb), lw=1, c='y')
    if (x[7]>0): plt.setp(lineab, linestyle='--')
    plt.gca().add_line(lineab)
    line0a = plt.Line2D((0,xa), (0,ya), lw=1, c='y')
    if (x[6]>0): plt.setp(line0a, linestyle='--')
    plt.gca().add_line(line0a)
    linebc = plt.Line2D((xb,xc), (yb,yc), lw=1, c='y')
    if (x[8]>0): plt.setp(linebc, linestyle='--')
    plt.gca().add_line(linebc)
    plt.axis('scaled')
    plt.show()

def F(x, f):  # gives you the array, of updated equations of forces that need to be maintained
    f[0] = L1*x[3]  +  L2*x[4]  +  L3*x[5]  -  L
    f[1] = L1*x[0]  +  L2*x[1]  -  L3*x[2]
    f[2] = x[6]*x[0]  -  x[7]*x[1]  -  W1
    f[3] = x[6]*x[3]  -  x[7]*x[4]
    f[4] = x[7]*x[1]  +  x[8]*x[2]  -  W2
    f[5] = x[7]*x[4]  -  x[8]*x[5]
    f[6] = pow(x[0], 2)  +  pow(x[3], 2)  -  1.0
    f[7] = pow(x[1], 2)  +  pow(x[4], 2)  -  1.0
    f[8] = pow(x[2], 2)  +  pow(x[5], 2)  -  1.0

#Calculate a _matrix_ of derivitives of df_i/d_j
def dFi_dXj(x, deriv, n):       # Define derivative function, using central difference
    h = 1e-3 #step size
    for j in range(0, n):
         temp = x[j]
         x[j] = x[j] +  h/2.
         F(x, f)                                                 
         for i in range(0, n): deriv[i, j] = f[i] #forward half-step
         x[j] = temp         
    for j in range(0, n):
         temp = x[j]
         x[j] = x[j] - h/2.
         F(x, f)
         for i in range(0, n): deriv[i, j] = (deriv[i, j] - f[i])/h  # (forward half-step - backward half-step) / step
         x[j] = temp

for it in range(1, 1000):
    F(x, f)                              
    dFi_dXj(x, deriv, n)   
    #  F' dx = -f
    sol = np.linalg.solve(deriv, -f) # used to solve for delta x
    
    x=x+sol

    errX = errF = errXi = 0.0
    willbreak=False
    eps = 1e-10
    for i in range(0, n):
        if ( x[i] !=  0.): errXi = abs(sol[i]/x[i])
        else:  errXi = abs(sol[i])
        if ( errXi > errX): errX = errXi                            
        if ( abs(f[i]) > errF ): errF = abs(f[i])        
        if ( (errX <=  eps) and (errF <=  eps) ): willbreak=True
    if (willbreak): break

plotconfig() #draw it
F(x,f) #evaluate solution
print('Number of iterations = ', it)
print('Solution:')
for i in range(0, n):
    print('x['+str(i)+']='+str(x[i]), 'f['+str(i)+']='+str(f[i]))
