from pylab import *
underf =1.
overf = 1.
underi =-1 # int
overi = 1 # int

N =10000
prevover = over
for i in xrange(0,N):
    underf = underf/2
    if(type(underf) != float):
        print("underflow for floats at %i \n" %i)
        break
    if(underf == 0):
        print('underflow for floats at %d \n' %i)
        break
print("done with underflow for float %f \n" %underf) 

underf =1.
overf = 1.
underi =-1 # int
overi = 1 # int

for i in xrange(0,N):
    underi = underi*2
    if(type(underi) != int):
        print("underflow for integers at %i \n" %i)
        break
    if(underi > 0):
        print('underflow for integers at %d \n' %i)
        break
#print("done with under integer %d \n" %underi) 
print("done with underflow for integers %i \n" %(underi/2)) 

underf =1.
overf = 1.
underi =-1 # int
overi = 1 # int

for i in xrange(0,N):
    prevover = overf
    overf = overf *2
    if(prevover > overf):
        print("overflow for floats at %i \n" %i)
        break 
    if(overf == inf):
        print("overflow for float at %i \n" %i)
        break
print("done with over flow for floats: %f \n" %(overf/2)) 

underf =1.
overf = 1.

underi =-1 # int
overi = 1 # int


for i in xrange(0,N):
    prevover = overi
    overi = overi *2
    if(type(overi) != int):
        print("overflow for integers at %i \n" %i)
        break
print("done with overflow for integers %f \n" %(overi/2)) 