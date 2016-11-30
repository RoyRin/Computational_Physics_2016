import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as p
from numpy import sin,cos,exp,pi
from mpl_toolkits.mplot3d import Axes3D

steps = 100
iterations = 100
res = 50.
pots = []
charge = []
fixed = [] # parallel array saying which values are fixed
for i in range(steps):
    pots.append([])
    fixed.append([])
    charge.append([])
    for j in range(steps):
        charge[-1].append(0.)
        pots[-1].append(0.)
        fixed[-1].append(0)

# test first with one charged side
for i in range(len(pots[0])):
    fixed[0][i] = 1
    fixed[steps-1][i] = 1
    fixed[i][0] = 1
    fixed[i][steps-1] = 1

# capacitor goes from 20 to 80, at 40 and 60
pluspos = range(38,43)
negpos = range(58,63)
plus = 100.
minus = -100.
caprange = range(20,81)
for i in caprange:
    for p in pluspos:
        fixed[p][i] = 2
        pots[p][i] = plus
    for n in negpos:
        fixed[n][i] = 2
        pots[n][i] = minus

# only iterate from 1 to len(arr)-2 (avoiding outside edges)
# first of all, they're fixed, and secondly, it makes the algorithm way easier
for curiter in range(iterations):
    for i in range(1,len(pots)-1):
        for j in range(1,len(pots[i])-1):
            if fixed[i][j] == 0:
                pots[i][j] = .25*(pots[i+1][j]+pots[i-1][j]+pots[i][j+1]+pots[i][j-1])

for i in range(1,len(pots)-1):
    for j in range(1,len(pots[i])-1):
        if fixed[i][j] == 2:
            charge[i][j] = (pots[i][j] - .25*(pots[i+1][j]+pots[i-1][j]+pots[i][j+1]+pots[i][j-1]))/(pi)

plot_charge_color = False
plot_charge_wire = True
plot_map = False
plot_wire = False

if plot_charge_color:
    plt.imshow(charge)
    plt.title('Charge Distribution')
    plt.colorbar()
    plt.show()

if plot_charge_wire:
    X = []
    Y = []
    # this is bad
    for i in range(steps):
        X.append([])
        Y.append([])
        for j in range(steps):
            X[-1].append(i)
            Y[-1].append(j)
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.plot_wireframe(X,Y,charge,cstride=2,rstride=2)
    plt.show()

if plot_map:
    plt.imshow(pots)
    plt.title('Potential')
    plt.colorbar()
    plt.show()

# first one's in order
# next one's all the same
if plot_wire:
    X = []
    Y = []
    # this is bad
    for i in range(steps):
        X.append([])
        Y.append([])
        for j in range(steps):
            X[-1].append(i)
            Y[-1].append(j)
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.plot_wireframe(X,Y,pots,cstride=3,rstride=3)
    plt.show()

print 'done!'