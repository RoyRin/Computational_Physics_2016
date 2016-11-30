from pylab import *
from numpy import *
from numpy.linalg import *

import math 
import random

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from  datetime import datetime
import time
xsize = ysize = 10**3
#mat = np.random.rand(xsize,ysize)
mat = np.full((xsize,ysize),3.3)
#mat2 = np.random.rand(xsize,ysize)
mat2=  np.full((xsize,ysize),3.9)
#print 


i = 345
j = 456

t1 = time.time()
#perform the evaluation
ABij = 0
for k in xrange(0,ysize):
    ABij += dot(mat[i][k], mat2[k][j])
t2 = time.time()
print("ABij = %f" %ABij)
print('for the regular summation, the time is t2-t1 = %f' %(t2-t1))

t1 = time.time()
ABij = 0
#slicedij = dot(mat[i,:], mat2[:,j])
print("ABij through slicing is %f" %dot(mat[i,:], mat2[:,j]))
t2 = time.time()
print('for the sliced summation, the time is t2-t1 = %f' %(t2-t1))