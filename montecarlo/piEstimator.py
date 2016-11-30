from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import math 
import matplotlib.mlab as mlab
import random


inpi=0

N=10**7
for i in xrange(N):
    x = random.uniform(0,1)
    y = random.uniform(0,1)
    r = math.sqrt(x**2 + y**2)
    if (r< 1.):
            inpi += 1.0
mypi = inpi/(N*1.0)
print("pi value is %f \n" %(mypi*4))
print("poisson error : %f" %(1/math.sqrt(N))) 