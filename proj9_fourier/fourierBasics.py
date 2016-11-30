from numpy import *
import pylab as p
import numpy as np
import matplotlib.pyplot as plt 

#import scipy.fftpack


def f(t,w):
    return (3*np.cos(w*t*2*np.pi) + 2*np.cos(3.* w*t*2*np.pi) + np.cos(5.*w*t*2*np.pi))
#np.fft(y)

def f2(t,w):
    return (1*np.cos(w*t*2*np.pi) + 2*np.cos(3.* w*t*2*np.pi) + 3*np.cos(5.*w*t*2*np.pi))
    
# Number of samplepoints of the signal
N = 2**10
print N
# sample spacing

#x = np.linspace(0.0, N*T, N)
xmax = 4.*np.pi
#T = xmax/N
#T  = 500 # number of samples taken in the 
x = np.arange(0, xmax , xmax/N) 
#x = np.arange(0., N *1., 1.)#, xmax/N) # the function
#w = 2./(2.*np.pi)
w = 1. #2.*np.pi
#y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
#y = f(x,w)
y = f2(x,w)
yf = fft.fft(y) / (N/2.) # the fourier of it
yback = fft.ifft(yf) *N/2
print yback
print sum(abs(yf.real))
print sum(abs(yf.imag))
#xf = np.arange(0., N *1., 1.)
xf = np.linspace(0., xmax, N) # the k space of the fourier

#fig, ax = plt.subplots()
#ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))

fig1 = plt.figure(1)
#plt.plot(x,y)
print len(xf)
print len(yf)
plt.title("fourier decomposition")
plt.ylabel("amplitude")
plt.xlabel("frequency")
plt.plot(x[:N/2],y[:N/2], label = "original signal")
plt.plot(x[:N/2],yback[:N/2], label = "inverse FFT of FFT signal")
plt.legend(bbox_to_anchor=(0.95, .95), loc='upper right', borderaxespad=0.)
power = yf [:]
for i in xrange(0,len(power)):
    power[i] = yf[i].real**2 + yf[i].imag**2
    
#plt.plot(xf[:N/2],yf[:N/2])
#plt.plot(xf[:N/2], power[:N/2])
fig1.show()