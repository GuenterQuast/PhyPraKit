#! /usr/bin/env python
"""
script: test_mFit2   Fitting with iminiut

This example illustrates the special features of iminuit:
 - definition of a custom cost function 
      used to implement least squares method with correlated errors   
 - profile likelihood for asymmetric errors
 - plotting of profile likeliood and confidence contours

  supports iminuit vers. < 2.0 and >= 2.0

.. moduleauthor:: Guenter Quast <g.quast@kit.edu>
"""

from __future__ import print_function  # for python2.7 compatibility

# imports
from PhyPraKit import generateXYdata, mFit
import numpy as np, matplotlib.pyplot as plt

# -- the model function
def model(x, a=0.1, b=1., c=1.):
   return a*x**2 + b*x + c

# uncertainties 
sigy_abs = 0.1 # abs. independent errors on y
sigy_rel = 0.05 # relative error on y
syabscor=0.1  #  an absolute, correlated error on y
sigx_abs = 0.1 # absolute error on x 
sxrelcor=0.02  #  a relative, correlated error on x 

# generate the data
xmin =  1.
xmax =  10.
xdata=np.arange(xmin, xmax+1. ,1.)
nd=len(xdata)
mpars=[0.3, -1.5, 0.5]
np.random.seed(31415)
xt, yt, ydata = generateXYdata(xdata, model, sigx_abs, sigy_abs, 
  srely=sigy_rel, xrelcor=sxrelcor, yabscor=syabscor, 
  mpar=mpars )

# fit with iminuitFit
par, pare, cor, chi2 = mFit(model,
    xdata, ydata,       # data and uncertainties
    sx=sigx_abs,        # indep. x abs.
    sy=sigy_abs,        # indep. y abs.
    srelx=None,         # indep. x corr.
    srely=sigy_rel,     # indep. y corr.
    xabscor=None,       # correlated x abs.
    xrelcor=sxrelcor,   # correlated x rel.
    yabscor=syabscor,   # correlated y abs.
    yrelcor=None,       # correlated y rel.
    p0=(0.5, -1., 1.),  # initial guess for parameter values
    use_negLogL=True,  # use corrected chi2                        
    plot=True,          # show plot, options below  
    plot_cor=True       # show profiles and contours
    )                      

print('*==* fit with iminuit completed')
print('*==* data set')
print('  x = ', xdata)
print('  sx = ', sigx_abs)
print('  y = ', ydata)
print('  sy = ', sigy_rel*model(xdata, *par))
print('*==* fit result:')
print("  -> chi2:         %.3g"%chi2)
np.set_printoptions(precision=3)
print("  -> parameters:   ", par)
np.set_printoptions(precision=2)
print("  -> uncertainties:", pare) 
print("  -> correlation matrix: \n", cor) 
# show figures
plt.show()
