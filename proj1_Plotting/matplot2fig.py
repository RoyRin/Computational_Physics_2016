
from pylab import *                                     # Load matplotlib

Xmin = -5.0;      Xmax =  5.0;        Npoints= 500
DelX= (Xmax-Xmin)/Npoints                                       # Delta x
x1 = arange(Xmin, Xmax, DelX)                                  # x1 range
x2 = arange(Xmin, Xmax, DelX/20)                     # Different x2 range
y1 =  -sin(x1)*cos(x1*x1)                                    # Function 1
y2 =   exp(-x2/4.)*sin(x2)                                   # Function 2
                                                                                 
#        Figure 1 
figure(1)
subplot(2,1,1)                              # 1st subplot in first figure
plot(x1, y1, 'r', lw=2) 
xlabel('x');      ylabel( 'f(x)' );      title( '-sin(x)*cos(x^2)' )
grid(True)                                                    # Form grid
subplot(2,1,2)                              # 2nd subplot in first figure
plot(x2, y2, '-', lw=2)
xlabel('x')                                                 # Axes labels
ylabel( 'f(x)' )
title( 'exp(-x/4)*sin(x)' )

#        Figure 2 
figure(2)
subplot(2,1,1)                                # 1st subplot in 2nd figure
plot(x1, y1*y1, 'r', lw=2) 
xlabel('x');        ylabel( 'f(x)' );    title( 'sin^2(x)*cos^2(x^2)' )                                                                                                                    # form grid
subplot(2,1,2)                                # 2nd subplot in 2nd figure
plot(x2, y2*y2, '-', lw=2)
xlabel('x');       ylabel( 'f(x)' );      title( 'exp(-x/2)*sin^2(x)' ) 
grid(True)

show()                                                      # Show graphs