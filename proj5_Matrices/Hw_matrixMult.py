from pylab import *
from numpy import *
from numpy.linalg import *

import math 
import random

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

mat = array([[4,-2,1],[3,6,-4],[2,1,8]])
matI = inv(mat)

print(mat)
print(matI)
print(dot(mat, matI))
print "==============="
print(matI)
exactMatI = (1/263.) *array([[52,17,2],[-32,30,19],[-9,-8,30]])


print exactMatI
print ( matI-exactMatI)

print "problem 2"
b = array([[12,-25,32], [4,-10,22],[20,-30, 40]])
bT = transpose(b) # transposed array
x = dot(matI, bT)

print x
print "problem 3"
A = array([[1,1],[-1,1]])
w,v = eig(A) # w is the values, v is the vectors
print "values" 
print w
print "vectors" 
print v


----