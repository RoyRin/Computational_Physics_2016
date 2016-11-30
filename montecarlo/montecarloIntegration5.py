from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import math 
import matplotlib.mlab as mlab
import random


#montecarlo integral for 10th- integral (all evaluated 
#from 0 to 1 (for Xi) of (X1+x2...X10)^2 dx1 dx2 etc

N = 100000
x = [None]*10
summer = 0
integrand =0
for j in xrange(N):
    for i in xrange(10):
        x[i] = random.uniform(0,1)
    integrand = np.sum(x)**2
    summer = summer + integrand
print(summer/(N*1.0))


#subtraction method - 5.18
## what if function has singularities? like a delta function
# find a similar looking function (g(x) ( like a steep gausian) - then calculat the integral
# calculate integral g(x) analytically, calculate integral f(x) - (g(x) using  uniform montecarlo integration
# integral f(x) dx = integral (f(x)-g(x) ) dx   - integral g(x)
# integral you want = something flat, can be calculated by montecarlo uniform   - analytic integral



#variance method
# find a similar looking functoin w(x), then use w(x) as a probability distribution to be used for 
#the random samples - this is then weighted by 1/w(x)
#integral f(x) dx = integral dx w(x) * f(x) / w(x)  

# f(x) * w(x) is the same as evaluating f(x) using a random probability distribution of w(x)
# dividing by w(x) then accounts for the weighting (probability of that happening) 

# so, for 
# integral is  1/N (sum(f(x| evaluated across distribution w(x) ) / w(x)


    