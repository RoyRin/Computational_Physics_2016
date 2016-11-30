import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation


# Make some plots
#    Make a 2 functions on the same plot, with a legend, 
        #with error bars

Xmin = -5.0;      Xmax =  5.0;        Npoints= 500
DelX= (Xmax-Xmin)/Npoints                                       # Delta x
x1 = np.arange(Xmin, Xmax, DelX)                                  # x1 range
x2 = np.arange(Xmin, Xmax, DelX/20)                     # Different x2 range
def f1(x,f):
   return np.sin(x*f) 
def f2(x,f):
    return np.cos(x*f)
                         
fig1 = plt.figure(1)
fig1.add_subplot(2,1,1)
plt.plot(x1, f1(x1,3))
plt.subplot(2,1,2)
plt.plot(x2, f2(x2,3), 'r--')
fig1.suptitle("hello")
fig1.show()


fig2 = plt.figure(2)
plt.subplot(1,1,1)
plt.plot(x2, f2(x2,3), 'r--')
plt.title("hello")
fig2.show()

#     Make 2 sets of data points on the same plot, with line of best fit
d1 = [4,2,1,6,3,4,32,3]
d2 =np.arange(0,len(d1),1)

fig3 = plt.figure(3)
plt.subplot(111)
plt.plot(d2,d1, 'bo')

plt.suptitle("data points")
fig3.show()


#    make 3d plot   

x1 = np.arange(-5,5,0.1)
x2 = x1/3
x, y = np.meshgrid(x1,x2)
z = np.sin(x*y)
#h = plt.contour(x,y,z)
fig4 = plt.figure(4)
ax = fig4.gca(projection='3d')
ax.plot_surface(x,y,z)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.suptitle("sin(x*y) ")
fig4.show()

    # initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return line,
    animation 
def animate(i):
    x = np.linspace(0, 2, 1000)
    y = np.sin(2 * np.pi * (x)* 0.01 * i)
    line.set_data(x, y)
    return line,

# call the animator.  blit=True means only re-draw the parts that have changed.


fig = plt.figure()#makes the figure
ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))#this sets the plot
line, = ax.plot([], [], lw=2)#What does this do?<<<<<---------??????????????
anim = animation.FuncAnimation(fig, animate, init_func=None,
                               frames=200, interval=20, blit=True)
plt.suptitle("animation of sin graph")
plt.show()
    
#     make a manipulate, plot
#   fourier series with different number of frequencies
