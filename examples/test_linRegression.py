#! /usr/bin/env python
''' test linear regression with errors in x an y with odFIT
    (using ODR package from scipy)
 
..  author:: Guenter Quast <g.quast@kit.edu>
'''

from __future__ import print_function  # for python2.7 compatibility

import numpy as np, matplotlib.pyplot as plt
from PhyPraKit import generateXYdata, odFit

def fitf(x, a, b):     
  ''' linear model 
  '''   
  return a*x + b

# -- define test configuration 
xmin =  1.
xmax =  10.
#  the model y(x) 
a_true = 0.3
b_true = 1.0
def model(x):
   return a_true*x + b_true
  
# set some uncertainties
sigx_abs = 0.2  # absolute error on x 
sigy_rel = 0.05 # relative error on y

xdata=np.arange(xmin, xmax+1. ,1.)
nd=len(xdata)

# generate pseudo data
xt, yt, ydata = generateXYdata(xdata, model, sigx_abs, 0., srely=sigy_rel)
ex=sigx_abs * np.ones(nd) # set array of x errors
ey=sigy_rel* yt * np.ones(nd) # set array of y errors

# (numerical) linear regression
par, pare, cor, chi2 = odFit(fitf, 
  xdata, ydata, ex, ey, p0=(1., 1.))

# print data and fit result, show plot
print('*==* input data')
print(('  x = ', xdata))
print(('  sx = ', ex))
print(('  y = ', ydata))
print(('  sy = ', ey))
print('fit result')
print(('  a=%.2f+-%.2f, b=%.2f+-%.2f, corr=%.2f, chi2/df=%.2f\n'\
% (par[0], pare[0], par[1], pare[1], cor[0,1], chi2/(nd-2.))))
fig0=plt.figure('1st data set', figsize=(5., 5.))
plt.errorbar(xdata, ydata, xerr=ex, yerr=ey, fmt='rx' )
xplt=np.linspace(xmin, xmax,100) 
plt.plot(xplt, par[0]*xplt+par[1], 'g-')
plt.xlabel('x')
plt.ylabel('y')
      
plt.show()
