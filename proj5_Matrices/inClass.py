from numpy import *
from numpy.linalg import *

A = array([ [1,2,3] , [22,23,24] , [55,66,100]])
x = array ([1,2,3])
b = dot(A,x)

print ( dot(inv(A),b))

print A
print "asdasdasda"
Es, EV = eig(A)
print Es # eigen value
print EV # eigen vector
print "===="
print eig(A)
EVT = EV.T

print dot(A,EVT[0])
print Es[0]*EVT[0]