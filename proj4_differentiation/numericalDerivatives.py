from pylab import *
import numpy as np
import math 
import random
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import matplotlib.mlab as mlab


x1 = np.arange(0,10,0.01)

def FD(func, t,h): #foraard differentiation
    return (func(t + h) - func(t))/h
#def FDE(func
    
def CD(func, t,h): # central differntiation
    return (func(t + h/2) - func(t - h/2))/h
    
def ED(func, t,h): # extrapolated differentiation
    return (8* (func(t + h/4) - func(t-h/4))- (func(t+h/2)- func(t-h/2)))/3/h


print FD(np.sin, 1, 0.01)
print np.cos(1)
def f1(x): # The integrand
    return np.cos(x)

def f2(x):
    return exp(x)
def derf1(x):
    return -np.sin(x)
def derf2(x):
    return exp(x)
    # note that these derivatives are only good for taking hte derivative of analytic functions, not discretized questions
def secondDerivativeCD22(func,t,h):
    return ((func(t+h) -func(t))-(func(t) - func(t-h)))/h**2
def secondDerivativeCD23(func,t,h):
    return (func(t+h) + func(t-h)-2*func(t))/h**2
    
def relativeError(evaluated, x, exact,h):
    return abs((exact(x)-evaluated)/h)
def absoluteError(evaluated, x, exact):
    return abs(exact(x)-evaluated)
t = {0.1,1.0,100}
h = 0.01
print FD(f1,1, h)
print "\n"
print "cosine Error======================="
print "forward Differentiation"
t = 0.1
print("t= 0.1: %f" %relativeError(FD(f1,0.1, h), 0.1,derf1,h ))
print("t= 1.0: %f" %relativeError(FD(f1,1.0, h), 1.0,derf1,h ))
print("t =100 : %f" %relativeError(FD(f1,100., h), 100.,derf1,h ))
t = 1.
print "central Differentiation"
print("t= 0.1: %f" %relativeError(CD(f1,0.1, h), 0.1,derf1,h ))
print("t= 1.0: %f" %relativeError(CD(f1,1.0, h), 1.0,derf1,h ))
print("t =100. : %f" %relativeError(CD(f1,100., h), 100.,derf1,h ))
t = 100.
print "Extrapolated Differentiation"
print("t= 0.1: %f" %relativeError(ED(f1,0.1, h), 0.1,derf1,h ))
print("t= 1.0: %f" %relativeError(ED(f1,1.0, h), 1.0,derf1,h ))
print("t =100: %f" %relativeError(ED(f1,100., h), 100.,derf1,h ))

print "Exp Error======================="
print "forward Differentiation"
t = 0.1
print("t= 0.1: %f" %relativeError(FD(f2,0.1, h), 0.1,derf2,h ))
print("t= 1.0: %f" %relativeError(FD(f2,1.0, h), 1.0,derf2,h ))
print("t =100 : %f" %relativeError(FD(f2,100., h), 100.,derf2,h ))
t = 1.
print "central Differentiation"
print("t= 0.1: %f" %relativeError(CD(f2,0.1, h), 0.1,derf2,h ))
print("t= 1.0: %f" %relativeError(CD(f2,1.0, h), 1.0,derf2,h ))
print("t =100. : %f" %relativeError(CD(f2,100., h), 100.,derf2,h ))
t = 100.
print "Extrapolated Differentiation"
print("t= 0.1: %f" %relativeError(ED(f2,0.1, h), 0.1,derf2,h ))
print("t= 1.0: %f" %relativeError(ED(f2,1.0, h), 1.0,derf2,h ))
print("t =100: %f" %relativeError(ED(f2,100., h), 100.,derf2,h ))



#print relativeError(ED(f1,100., h), 100.,derf1,h )


#plt.plot(np.log10(relativeError(ED(f2,100., h), 100.,derf2,h )))


#second derivitive thing
h =0.1
x1 = np.arange(0,2*np.pi,0.001)
y1 = f1(x1)
xh1 = np.arange(0,15)
h1 = np.pi/(10 *10**xh1)

loc = 0.5
secDer22 = secondDerivativeCD22(f1,loc, h)
secDer23 = secondDerivativeCD23(f1,loc, h)
fig1 = plt.figure(1)
#plt.plot(y1,label = "cosine")
#def relativeError(evaluated, x, exact,h):
#    return abs((exact(x)-evaluated)/h)
h=0.1
#secDer = secondDerivativeCD22(f1,loc, h1)

loc = np.arange(0,2*np.pi, 1.)
for i in xrange (len(loc)):
    plt.plot(xh1,log10(relativeError(-secondDerivativeCD23(f1,loc[i], h1), loc[i], f1,h1)), label = "location %f " %loc[i])

def totalError(func, x, array,h): # returns the total error across x, for a function (just sum the relative error at each point)
    error  =0.0
    for i in xrange (len(array)):
        error += relativeError(-array[i],x,func ,h)
    return error
#print totalError(f1, x1,secDer
legend(bbox_to_anchor=(0.55, .45), loc=2, borderaxespad=0.)
plt.xlabel("Log of h (step size), (base 10)")
plt.ylabel(" Log of Error (base 10)")
plt.title("log log Error of Second derivative (eq.5.23) versus h (step size) ")
fig1.show()



#CD,FD,ED things
x = np.arange(0,15,0.3)
h = 1/(10.**x)
t = 1.0
fig2 = plt2.figure(2)
plt2.plot(x,log10(relativeError(ED(f1,t, h), t,derf1,h)), label = "Extrapolated Difference")
plt2.plot(x,np.log10(relativeError(FD(f1,t, h), t,derf1,h )), label = " Forward Difference")
plt2.plot(x,np.log10(relativeError(CD(f1,t, h), t,derf1,h )), label = "Central Difference")
legend(bbox_to_anchor=(0.65, .85), loc=2, borderaxespad=0.)
plt2.xlabel("-log h (base 10)")
plt2.ylabel("log Error (base 10)")
plt2.title("Error vs size of h @ t = %f" %t)
fig2.show()
