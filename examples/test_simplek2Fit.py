#! /usr/bin/env python
'''test_k2Fit

   test fitting simple line with kafe2, without any errors given

.. moduleauthor:: Guenter Quast <g.quast@kit.edu>
'''

from __future__ import print_function  # for python2.7 compatibility

# import kafe2 # must be imported first to properly set matplotlib backend
from PhyPraKit import generateXYdata, k2Fit
import numpy as np, matplotlib.pyplot as plt

# -- the model function
def model(x, a=1., b=0):
   return a*x + b 

# parameters for the generation of test data
xmin =  1.
xmax =  10.
xdata=np.arange(xmin, xmax+1. ,1.)
ydata=[1.1, 1.9, 2.95, 4.1, 4.9, 6.2, 6.85, 8.05, 8.9, 10.15]
ey=None
# fit with kafe2
par, pare, cor, chi2 = k2Fit(model, xdata, ydata, sy=ey)

print('*==* data set')
print('  x = ', xdata)
print('  y = ', ydata)
print('  sy = ', ey)
print('*==* fit result:')
print("  -> chi2:         %.3g"%chi2)
np.set_printoptions(precision=3)
print("  -> parameters:   ", par)
np.set_printoptions(precision=2)
print("  -> uncertainties:", pare) 
print("  -> correlation matrix: \n", cor) 
