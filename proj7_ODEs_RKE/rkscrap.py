#4th order Runge Kutta
#Equation: dy/dx = 3*exp(-x) - 0.4y; initial conditions: y(0) = 5, X(0) = 0, h = 1.5
#exact solution y = 10*exp(-0.4x) - 5*exp(-x)
import numpy, math, matplotlib
from pylab import *
x0, y0, h, n = 0, 5, 1.5, 10 #initial conditions, step size, number of steps
a = [y0] #place y0 in array
for i in arange(0,n,1):
    k1x, k1y = x0 + i*h, y0
    k1 = 3*exp(-1*k1x) - 0.4*k1y
    k2x, k2y = x0 + i*h + 0.5*h, y0 + 0.5*k1*h
    k2 = 3*exp(-1*k2x) - 0.4*k2y
    k3x, k3y = x0 + i*h + 0.5*h, y0 + 0.5*k2*h
    k3 = 3*exp(-1*k3x) - 0.4*k3y
    k4x, k4y = x0 + i*h + h, y0 + k3*h
    k4 = 3*exp(-1*k4x) - 0.4*k4y
    y1 = y0 + (1/6.0)*(k1 + 2*k2 + 2*k3 + k4)*h
    y0 = y1
    a.append(y1)
print("length of a is %d" %len(a))

x = np.arange(0,n+1,1)
print("length of x is %d" %len(x))

fig3 = plt.figure(3)
plt.subplot(111)
plt.plot(x,a, 'bo')

plt.suptitle("data points")
fig3.show()