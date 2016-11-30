""" From "COMPUTATIONAL PHYSICS", 3rd Ed, Enlarged Python eTextBook  
    by RH Landau, MJ Paez, and CC Bordeianu
    Copyright Wiley-VCH Verlag GmbH & Co. KGaA, Berlin;  Copyright R Landau,
    Oregon State Unv, MJ Paez, Univ Antioquia, C Bordeianu, Univ Bucharest, 2015.
    Support by National Science Foundation"""

# DFTcomplex.py:  Discrete Fourier Transform, using built in complex
from pylab import *
import matplotlib.pyplot as plt
import time

power = 10
N = 2**power;  Np = N                           # Number points
signal = zeros( (N+1), float )     
twopi  = 2.*pi;       sq2pi = 1./sqrt(twopi);         h = twopi/N
dftz   = zeros( (Np), complex )               # sequence complex elements

def f(signal):                                          # signal function
    step = twopi/N;        x = 0. 
    for i in range(0, N+1):
       signal[i] = 1.0+1.0*cos(x)
       x += step
      
def fourier(dftz):                                                  # DFT
    for n in range(0, Np):              
      zsum = complex(0.0, 0.0)               # real and imag parts = zero
      for  k in range(0, N):                              # loop for sums
          zexpo = complex(0, twopi*k*n/N)              # complex exponent
          zsum += signal[k]*exp(-zexpo)          # Fourier transform core
      dftz[n] = zsum * sq2pi                                     # factor
      #if n>Np/2: dftz[n] = 0
      #print dftz[n]

f(signal)                       # call signal
start = time.clock()
fourier(dftz)                   # transform
end = time.clock()
print 'DFT took', end - start,'seconds'

plt.figure(1)
plt.plot(signal)
plt.plot(dftz.real*5./N/2.,"bo")
plt.plot(dftz.imag*5./N/2.,"ro")
#plt.xkcd()
#plt.rcdefaults()
plt.show()
