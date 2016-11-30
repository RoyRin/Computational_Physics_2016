from pylab import *
import numpy as np
import matplotlib.pyplot as plt

maxTerms = 50

def sineFunc(x, n):
    terms = []
    terms.append(x)
    for i in range(1,n):
       #terms.append(0)
       terms.append(terms[i-1]*(-1*(x**2)/(((2*n)-1)*((2*n)-2)) )  )
    return terms


print("this is the sum of sin(pi/2) %d" %sum(sineFunc(pi/2,10) ))


def summer(arr, n):
    sum = 0
    if len(arr)>=n:
        for i in range(0,n):
            sum = sum+arr[i]
    return sum
    

Xmin = -pi;      Xmax =  pi;        Npoints= 100
DelX= (Xmax-Xmin)/Npoints                                       # Delta x
x1 = np.arange(Xmin, Xmax, DelX)                                  # x1 range




summedTerms = []
for i in xrange(len(x1)):
    summedTerms.append(sum(sineFunc(x1[i],10)))

plt.plot(x1,summedTerms)
plt.show()

#termsMat =[[0 for x in range(len(x1))] for y in range(maxTerms)] 
termsMat =[0 for x in range(len(x1))]  #terms mat will produce an array at each x value, 
                    #for the evaluated terms of the sin

for i in xrange(len(x1)):
    for j in xrange(maxTerms):
        termsMat[i] = sineFunc(x1[i], j) # terms mat creates an array of sin terms for each x value

def summedTermsMat(n):
    summedTermsMat =[]
    for i in xrange(0,len(termsMat)):
        summedTermsMat.append(summer(termsMat[i],n))
    return summedTermsMat
#print(len(x1))
#print(len(summedTermsMat(4)))
#print(summedTermsMat(4))     
                         
fig1 = plt.figure(1)
fig1.add_subplot(2,1,1)

#plt.plot(np.arange(maxTerms), summer(termsMat, [0:100]))
val = []

for j in xrange(0,len(x1)):
    val.append(summer(termsMat[j],maxTerms))
plt.plot(val)
    

#plt.plot(x1, summedTermsMat(10) )
#plt.subplot(2,1,2)
#plt.plot(x2, f2(x2,3), 'r--')

#fig1.suptitle("approximated values of Sin")
#plt.ylabel('approx values of sin')
#plt.xlabel('x')

#fig1.show()
