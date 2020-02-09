#! /usr/bin/env python
'''test_k2Fit

   test fitting an arbitrary fucntion with kafe2, 
   with uncertainties in x and y and correlated 
   absolute and relative uncertainties

.. moduleauthor:: Guenter Quast <g.quast@kit.edu>
'''

from __future__ import print_function  # for python2.7 compatibility

# import kafe2 # must be imported first to properly set matplotlib backend
from PhyPraKit import generateXYdata, k2Fit
import numpy as np, matplotlib.pyplot as plt

# -- the model function
def model(x, a=0.3, b=1., c=1.):
   return a*x**2 + b*x + c

# parameters for the generation of test data
sigx_abs = 0.2 # absolute error on x 
sigy_rel = 0.1 # relative error on y
#       errors of this kind only supported by kafe
sxrelcor=0.05 #  a relative, correlated error on x 
syabscor=0.1  #  an absolute, correlated error on y
xmin =  1.
xmax =  10.
xdata=np.arange(xmin, xmax+1. ,1.)
nd=len(xdata)
mpars=[0.3, -1.5, 0.5]

# generate the data
xt, yt, ydata = generateXYdata(xdata, model, sigx_abs, 0., 
  srely=sigy_rel, xrelcor=sxrelcor, yabscor=syabscor, 
  mpar=mpars )
ey=sigy_rel* yt * np.ones(nd) # set array of relative y errors

# fit with kafe2
par, pare, cor, chi2 = k2Fit(model,
    xdata, ydata, sigx_abs, ey,               # data and uncertaintites
    p0=None, p0e=None,                        # initial guess and range
    xrelcor=sxrelcor, yabscor=syabscor,       # correlated uncertainties
    plot=True,                                # show plot, options below  
    axis_labels=['x-data','random y'],        # nice names
    data_legend = 'random data',              # legend entry for data points
    model_name = 'quadratic',                 # name for model
    model_expression = r'{0}\,x^2 + {1}\,x + {2}',  # model fuction
    model_legend = 'quadratic model',         # legend entry for model line
    #model_band = None,                       # name for model uncertainty band
    fit_info=True )                           # fit results in figure

# setting any of the above names to None will remove the entry from the legend,
#  if not specified, use default  

print('*==* data set')
print('  x = ', xdata)
print('  sx = ', sigx_abs)
print('  y = ', ydata)
print('  sy = ', ey)
print('*==* fit result:')
print("  -> chi2:         %.3g"%chi2)
np.set_printoptions(precision=3)
print("  -> parameters:   ", par)
np.set_printoptions(precision=2)
print("  -> uncertainties:", pare) 
print("  -> correlation matrix: \n", cor) 

