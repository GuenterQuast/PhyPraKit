#! /usr/bin/env python
'''test_generateDate

     test generation of simulated data
     this simulates a measurement with given x-values with uncertainties;
     random deviations are then added to arrive at the true values, from
     which the true y-values are then calculated according to a model
     function. In the last step, these true y-values are smeared
     by adding random deviations to obtain a sample of measured values 

..  moduleauthor:: Guenter Quast <g.quast@kit.edu>

'''
import numpy as np, matplotlib.pyplot as plt
from PhyPraKit import generateXYdata

# define the true model
def model(x, a=1., b=0.):
   return a*x+b

#  define data points 
xmin=1.
xmax=10.
xm=np.arange(xmin, xmax+1., 1.)
nd=len(xm)
# set errors
sx = 0.15 * np.ones(nd)
sy = 0.2 * np.ones(nd)
xabscor = 0.07 # common error of 0.5 on x
yabscor = 0.  
xrelcor = 0.
yrelcor = 0.1  # common error of 0.1=10% on y

# generate the data:
#   assumes xm is measured, xt then are the true values where
#    the model is evaluated to obtain the measured values ym
xt, yt, ym = generateXYdata(xm, model, sx, sy, 
   xabscor=xabscor, yrelcor=yrelcor)

# compute total errors
stotx = np.sqrt(sx**2 + xabscor**2 + (xt*xrelcor)**2)
stoty = np.sqrt(sy**2 + yabscor**2 + (yt*yrelcor)**2)

# now plot what we got
fig, ax=plt.subplots(1,1)
#   1. the true model 
dx=(xmax-xmin)/10.  
xp=np.linspace(xmin-dx, xmax+dx, 200) 
ax.plot(xp, model(xp), 'g-')
#  2. the generated data
ax.errorbar(xm, ym, xerr=stotx, yerr=stoty, fmt='ro')  
ax.set_xlabel('measured values of x')
ax.set_ylabel('model(x)')

plt.show() # display on screen

