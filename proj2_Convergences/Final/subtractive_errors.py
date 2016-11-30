from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import csv


def s1(n):
    summed = 0
    for i in xrange(1,(2*n +1)):
        j= i *1.0
        term = (-1.0)**i * (j/(j+1.))
        summed = summed + term
    return summed
print(s1(2))

def s2(n):
    sum1 = 0
    sum2 = 0
    for i in xrange(1,(n +1)):
        j= i *1.0
        term = ((2*j)-1)/(2*j)
        sum1 = sum1 + term
        
    for i in xrange(1,(n +1)):
        j= i *1.0
        term = ((2*j))/(2*j +1)
        sum2 = sum2 + term
    return (-1*sum1 + sum2)
print(s2(2))

def s3(n):
    summed = 0.0
    for i in xrange(1,(n+1)):
        j= i *1.0
        term =((1.)/((2.*j)*(2.*j+1.)))
        summed = summed + term
    return summed
print(s3(2))


x0 = np.arange(1.,7.,1.)
x1 = 10.**x0
#x1 = np.arange(1,100000,100)
print(x1)
#x1 = [1,2,3]
fig1 = plt.figure(1)
fig1.add_subplot(1,1,1)
relErrorSum1 =[]

for i in xrange(len(x1)):
    if s3(int(x1[i])) !=0.:
        relErrorSum1.append(abs((s1(int(x1[i]))-s3(int(x1[i])))/(s3(int(x1[i])))))
    else: 
        relErrorSum1.append(abs((s1(int(x1[i])))/(0.001)))
relErrorSum2 =[]

for i in xrange(len(x1)):
    if s3(int(x1[i])) !=0.:
        relErrorSum2.append(abs((s2(int(x1[i]))-s3(int(x1[i])))/(s3(int(x1[i])))))
    else: 
        relErrorSum2.append(abs((s2(int(x1[i])))/(0.001)))

#plt.plot(x1,log10(relErrorSum1), x1, log10(relErrorSum2))
#plt.plot(x1, (relErrorSum1),label ="Relative error of sum 1")
#plt.plot(x1, (relErrorSum2),label ="Relative error of sum 2")
plt.plot(log10(x1), log10(relErrorSum1),label ="Relative error of sum 1")
plt.plot(log10(x1), log10(relErrorSum2),label ="Relative error of sum 2")



legend(bbox_to_anchor=(0.55, .25), loc=2, borderaxespad=0.)
plt.xlabel("Log of N value")
plt.ylabel("Log of Relative Error of Sums")
plt.title("Relative Error of Computing Sums")
fig1.show()         