""" From "COMPUTATIONAL PHYSICS", 3rd Ed, Enlarged Python eTextBook  
    by RH Landau, MJ Paez, and CC Bordeianu
    Copyright Wiley-VCH Verlag GmbH & Co. KGaA, Berlin;  Copyright R Landau,
    Oregon State Unv, MJ Paez, Univ Antioquia, C Bordeianu, Univ Bucharest, 2015.
    Support by National Science Foundation"""

# FFT.py:  FFT for complex numbers in dtr[][2], returned in dtr

from numpy import *
import matplotlib.pyplot as plt
import time

power=6
nn = 2**power                                      # Power of 2
max = 4*2**power
#max = 100
points = max                                          # Can be increased
data = zeros((max), float) 
dtr  = zeros((points,2), float)
signal = zeros( nn, float )  
fftreal = zeros( nn, float )
fftimag = zeros( nn, float )

def myfft(nn,isign):                                      # FFT of dtr[n,2]
    n = 2*nn
    for i in range(0,nn+1):                # Original data in dtr to data
         j = 2*i+1
         data[j] = dtr[i,0]                       # Real dtr, odd data[j]
         data[j+1] = dtr[i,1]                  # Imag dtr, even data[j+1]
    j = 1                               # Place data in bit reverse order
    for i in range(1,n+2, 2):
        if (i-j) < 0 :                # Reorder equivalent to bit reverse
            tempr = data[j]
            tempi = data[j+1]
            data[j] = data[i]
            data[j+1] = data[i+1]
            data[i] = tempr
            data[i+1] = tempi 
        m = n/2;
        while (m-2 > 0): 
            if  (j-m) <= 0 :
                break
            j = j-m
            m = m/2
        j = j+m;
                               
    #print(" Bit-reversed data ")  
    #for i in range(1, n+1, 2): print("%2d  data[%2d]  %9.5f "%(i,i,data[i]))    # To see reorder

    mmax = 2
    while (mmax-n) < 0 :                                # Begin transform
       istep = 2*mmax
       theta = 6.2831853/(1.0*isign*mmax)
       sinth = math.sin(theta/2.0)
       wstpr = -2.0*sinth**2
       wstpi = math.sin(theta)
       wr = 1.0
       wi = 0.0
       for m in range(1,mmax +1,2):  
           for i in range(m,n+1,istep):
               j = i+mmax
               tempr = wr*data[j]   -wi *data[j+1]
               tempi = wr*data[j+1] +wi *data[j]
               data[j]   = data[i]   -tempr
               data[j+1] = data[i+1] -tempi
               data[i]   = data[i]   +tempr
               data[i+1] = data[i+1] +tempi        
           tempr = wr
           wr = wr*wstpr - wi*wstpi + wr
           wi = wi*wstpr + tempr*wstpi + wi;
       mmax = istep              
    for i in range(0,nn):
        j = 2*i+1
        dtr[i,0] = data[j]
        dtr[i,1] = data[j+1] 

isign = -1                           # -1 transform, +1 inverse transform
#print('        INPUT')
#print("  i   Re part   Im  part")
def sig(i,phase):
 #   return i**3
    return 1.0+1.0*cos(3.*i*phase) 
    
for i in range(0,nn ):                                       # Form array
    phase=2.*pi/nn
    dtr[i,0] = sig(i,phase)                                         # Real part
    signal[i] = dtr[i,0]
    dtr[i,1] = 0.0                                            # Im part
    #print(" %2d %9.5f %9.5f" %(i,dtr[i,0],dtr[i,1]))

start = time.clock()
myfft(nn, isign)                             # Call FFT, use global dtr[][]
end = time.clock()

#print('    Fourier transform')
#print("  i      Re      Im    ")
for i in range(0,nn):
    #print(" %2d  %9.5f  %9.5f "%(i,dtr[i,0]/nn,dtr[i,1]/nn))
    fftreal[i]=dtr[i,0]/nn
    fftimag[i]=dtr[i,1]/nn

print 'FFT took', end - start,'seconds but DFT would take', 2*(end - start)*nn/log2(nn),'seconds'

plt.figure(0)
plt.plot(signal) # plot the signal itself

start = time.clock()
sp = fft.fft(signal) #numpy function
end = time.clock()
print 'numpy FFT took', end - start,'seconds'

plt.plot(sp.real/nn,"ys",ms=10) #numpy answer - yellow squares
plt.plot(fftreal,"bo") # real parts - blue circles
plt.plot(fftimag,"ro") # imaginary parts - red circles
plt.ylabel("amplitude")
plt.xlabel("frequency")
plt.title("FFT Analysis")
plt.show()