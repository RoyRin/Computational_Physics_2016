from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import csv

maxTerms = 50
epsilon = 10**-7
Xmin = -3*pi;      Xmax =  3*pi;        Npoints= 400
DelX= (Xmax-Xmin)/Npoints                                       # Delta x
x1 = np.arange(Xmin, Xmax, DelX)                                  # x1 range

def sineFunc(x, n):
    terms = []
    del terms[:]
    terms.append(x)
    for i in range(1,n):
       #terms.append(0)
       terms.append(terms[i-1]*(-1*(x**2)/(((2*n)-1)*((2*n)-2)) )  )
    return terms

def sineVal(x,e): #returns the value of the taylor of sin function, for a given epsilon
    term = x * 1.0
    summed = x* 1.0
    i =1.0
    while(abs(term/summed) > e):
        i = i +1.0
        term = term *(-1.*x*x) /((2.*i)-1.)/((2.*i)-2.)
        summed = term + summed 
    return summed
    
def sineValtoTerm(x,n): #returns the value of the taylor of sin function, up to some term
    term = x * 1.0
    summed = x* 1.0
    j =1.
    for i in xrange(n):
        j = j+1.0
        term = term *(-1.*x*x) /((2.*j)-1.)/((2.*j)-2.)
        summed = term + summed 
    return summed
def sineValTerm(x,t): #returns the term of the taylor of sin function
    j =1. * t
    print(j)
    term = ((-1.)**(j-1)*x**(2*j-1))/math.factorial((2*j)-1)
    return term
    
def sineValtoTermFactorial(x,n): #returns the value of the taylor of sin function, up to some term
    term = x * 1.0
    summed = x* 1.0
    j =1.
    for i in xrange(n):
        j = j+1.0
        term = ((-1.)**(j-1)*x**(2*j-1))/math.factorial(2*j-1)
        summed = term + summed 
    return summed

    
print("sine val pi/2) %f" %sineVal(pi/2, epsilon))
print("this is the sum of sin(pi/3) %f" %sum(sineVal(pi/3, epsilon) ))

sineArrayEpsilon = []#this is an array of an array of sine values, taken to some epsilon value of accuracy
for i in xrange(len(x1)):
    sineArrayEpsilon.append(sineVal(x1[i],epsilon))
    
def sineTerms(x,t):    
    Terms = []#this is an array of an array of sine values, taken to some epsilon value of accuracy
    for i in xrange(len(t)):
        Terms.append(sineValTerm(x,t[i]))
    return Terms
    
terms = 50
sineArrayTerm = [] #this is an array of an array of sine values, taken the nth term
for i in xrange(len(x1)):
    sineArrayTerm.append(sineValtoTerm(x1[i],terms))   

sineArrayFactorialTerm = []#this is an array of an array of sine values, taken to some epsilon value of accuracy
for i in xrange(len(x1)):
    sineArrayFactorialTerm.append(sineValtoTermFactorial(x1[i],terms)) 

#this section writes to a csv file
myfile = open("Dropbox/NYU/ComputationalPhysics/project2_Convergences/Final/myfile.txt", 'w+')
wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
wr.writerow(["evaluation of sin", "difference from sin", "relative error"])
for i in xrange(len(sineArrayEpsilon)):
    a = [sineArrayTerm[i], sineArrayTerm[i] - sin(x1[i]),(sineArrayTerm[i] - sin(x1[i])) / sin(x1[i])]
    wr.writerow(a)

nTerms = 200.#number of terms
rangeX = 100.# range of x values
xEvalPts = 1000  #number of points to evaluate
xEvalPlots = 10#number of plots to make of relative error (later)

xEval = np.arange(-rangeX,rangeX,(rangeX /xEvalPts))
sineConvergence = [] #this is an array of an array of sine values, taken the nth term
#it is iterated over the number of terms it takes(row-wise), and iterated over the values it is evaluated on,
#over the columns

for i in xrange(int(nTerms)):
    a= abs(sineValtoTerm(xEval,i) - sin(xEval)/(sin(xEval)))
    sineConvergence.append(a)   
    #creates a 2D matrix of values that represent the relative error at (a given x, for a given number of terms)
    
sineConvergenceT = [[sineConvergence[j][i] for j in range(len(sineConvergence))] for i in range(len(sineConvergence[0]))]
#sineConvergenceT is the transpose of sineConvergence - later to be used for easier plotting

# I want to plot, for a given x value, the number of terms required at the first convergence, for a given epsilon

delta = 0.1
ratioConvergence = []
breaker = 0
#---for i in xrange(int(nTerms)): # iterate for each array of sin values evaluations for a certain number of terms
 #  ---- for j in xrange(5,len(sineConvergenceT[i])): #iterate through the actual values of the evaluation
        # if the difference of relative errors of 2 values is less than delta:
        #print(abs(sineConvergence[i][j] - sineConvergence[i][j-5]))
       # if abs(sineConvergenceT[i][j] - sineConvergenceT[i][j-5])< delta:
            #ratioConvergence takes the value: xvalue / number of terms
  #---          ratioConvergence.append(abs(xEval[j]/(i*1.0)))
            #ratioConvergence.append(abs(xEval[j]))
           # print("j is at x distance %f" %xEval[j])
            #ratioConvergence.append(abs(sineConvergence[i][j] - sineConvergence[i][j-1]))
       #     break

for i in xrange(len(xEval)):# i tells you the number of terms involved
    c =xEval[i]**2 / delta
    ratioConvergence.append((-2+math.sqrt(4+16*c))/2)


#plot the growth and decay (importancea) of each term in the taylor series
#fig0 = plt.figure(0)
#fig0.add_subplot(1,1,1)
#numTerms= np.arange(1,40,1)
#exes= [0.1,1.0,2,10]
#plt.plot(exes, sineTerms(exes[1],numTerms))
#plt.xlabel("Term index")
#plt.ylabel("Sine Term evaluations")
#plt.title("Plot of Sin Taylor Terms" %terms)
#fig0.show()




fig1 = plt.figure(1)
fig1.add_subplot(3,1,1)
plt.plot(x1, sineArrayTerm)
plt.xlabel("x value")
plt.ylabel("Sine approximation")
plt.title("Plot of Sin Taylor series to the %d th term" %terms)
plt.subplot(3,1,3)
plt.plot(x1, sineArrayEpsilon)
plt.title("Plot of Sin Taylor series error of Epsilon : %s" %epsilon)
plt.xlabel("x value")
plt.ylabel("Sine approximation")
fig1.suptitle("Sin approximations arrays")
#fig1.show()

fig2 = plt.figure(2)
fig2.add_subplot(1,1,1)
plt.plot(x1, sineArrayFactorialTerm )
plt.xlabel("x value")
plt.ylabel("Sine approximation by factorial")
plt.title("Plot of Sin Taylor series to the %d th term" %terms)
#fig2.show()

fig3 = plt.figure(3)
#fig2.add_subplot(1,1,1)
ax = subplot(1,1,1)
#for i in xrange(len(xEval)):   
for i in xrange(xEvalPlots):
    j = i* (xEvalPts/xEvalPlots)
    stringA = str("xvalue %f" %xEval[i])
    plt.semilogy(log10(sineConvergenceT[j]), label=stringA)
    #this plots and labels the relative error of the series for a given x value (iterated over a series of x vals)

legend(bbox_to_anchor=(0.65, .85), loc=2, borderaxespad=0.)
plt.xlabel("Number of Terms")
plt.ylabel("Absolute Val of Log(base 10) of relative Errors")
plt.title("Plot of Error in the sin series, by term")
#fig3.show()

fig4 = plt.figure(4)
ax = subplot(1,1,1)

plt.plot(xEval, ratioConvergence)
plt.xlabel("x-Value")
plt.ylabel("terms required for Convergence for Epsilon = %f" %delta)
plt.title("Rate of Convergence for a given Epsilon")

fig4.show()
fig3.show()
fig2.show()            
fig1.show()
                        
            
            