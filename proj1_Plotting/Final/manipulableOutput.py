from pylab import *
from matplotlib.widgets import Slider, Button, RadioButtons

fig1 = figure(1)# make a figure
plt.suptitle("manipulatable interactive function")
ax = subplot(111)
subplots_adjust(left=0.25, bottom=0.25, right =0.9, top = 0.9) # sets the bottom and left borders - right and top are defaults
t = arange(0.0, 1.0, 0.001) # list of points from 0 to 1, 100
a0 = 5 # initial amplitude
f0 = 3 # initial frequency
def s0(a,f):
    return a*np.sin(f*t)
def s1(a,f):
    return a*np.cos(f*t)
def s2(a,f):
    return a*np.tan(f*t)
    
funcName = 0
signal= s0 #initial signal (it is an array of points, length of array t)

l, = plot(t,signal(a0,f0), lw=2, color='red') #initial plot
axis([0, 1, -10, 10]) #making the axis for the figure

axcolor = 'lightgoldenrodyellow'
axfreq = axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor) # set the specifics for the axes
axamp  = axes([0.25, 0.15, 0.65, 0.03], axisbg=axcolor)# set the specifics for the axes

#makes the frequency and amplitude slider
sfreq = Slider(axfreq, 'Freq', 0.1, 30.0, valinit=f0) # now comes the real part - making the slider
samp = Slider(axamp, 'Amp', 0.1, 10.0, valinit=a0) # make 2nd slider function, with initial functions
#makes the reset button
resetax = axes([0.8, 0.025, 0.1, 0.04]) # where the reset button is located
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975') # reset button
# makes the color radio button
rax = axes([0.025, 0.5, 0.15, 0.15], axisbg=axcolor) # sets radio button location
radioColor = RadioButtons(rax, ('red', 'blue', 'green'), active=0) # makes radio buttons for colors
# makes the function setting radio button
funcList = ('Sin', 'Cos', 'Tan')
raxFunc = axes([0.025, 0.8, 0.15, 0.15], axisbg=axcolor) # sets radio button location
axcolor = 'lightgoldenrodyellow'
radioFunc = RadioButtons(raxFunc, funcList) 

#slider updating functionality
def update(val):
    global funcName
    print "function %d", funcName
    amp = samp.val   # this takes the value that it has been set on the slider
    freq = sfreq.val # this takes the value that it has been set on the slider
    #ydata = trigdict[label](amp,freq) # sets the y data equal to correct function b/c label 
        #tells you the current status of the radio button
    signal = s0
    if funcName == 0:
        signal = s0  
    if funcName == 1:
        signal = s1   
    if funcName == 2:
        signal = s2
    print(funcName)    
    l.set_ydata(signal(amp,freq) ) # reupdates the array of amplitude values
    draw()  # redraw the plot, at the end of the update
sfreq.on_changed(update) # connects to the slider event, so when slider of sfreq is updated, it calls update
samp.on_changed(update) # connects to the slider event, so when slider of sfreq is updated, it calls update


# reset button functionality
def reset(event):
    sfreq.reset()
    samp.reset()
button.on_clicked(reset) # links resetting to the reset button press event


# color radio button functionality
def colorfunc(label): # 
    l.set_color(label) # colors
    draw() # redraws
radioColor.on_clicked(colorfunc)


# different function functionality  
def trigfunc(label):
    global funcName
    amp = samp.val   # this takes the value that it has been set on the slider
    freq = sfreq.val # this takes the value that it has been set on the slider
    trigdict = {'Sin':s0, 'Cos':s1, 'Tan':s2} # this is essentially an if 
        #statement that if the label is sin, then trig dict is s0
  #  signal = trigdict[label]
 
    if label == 'Sin':
        funcName = 0
        print("function is 0")
    if label == 'Cos':
        funcName = 1
        print("function is 1")
    if label == 'Tan':
        funcName = 2
        print("function is 2")
    print(label)
    ydata = trigdict[label](amp,freq) # sets the y data equal to correct function b/c label 
        #tells you the current status of the radio button
    l.set_ydata(ydata) # resets the data
    plt.draw() # redraws
    
radioFunc.on_clicked(trigfunc) # links the radio click event to the redrawing


show()
