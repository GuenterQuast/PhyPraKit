#! /usr/bin/env python
"""
test_k2Fit
   Illustrate fitting of an arbitrary function with kafe2  
   This example illustrates the special features of kafe2:
   - correlated errors for x and y data  
   - relative errors with reference to model
   - profile likelihood method to evaluate asymmetric errors
   - plotting of profile likeliood and confidence contours

.. moduleauthor:: Guenter Quast <g.quast@kit.edu>
"""

from __future__ import print_function  # for python2.7 compatibility

# import kafe2 # must be imported first to properly set matplotlib backend
from PhyPraKit import generateXYdata, k2Fit
import numpy as np, matplotlib.pyplot as plt

# -- the model function
def model(x, a=0.1, b=1., c=1.):
   return a*x**2 + b*x + c

if __name__ == "__main__": # --------------------------------------  

  # uncertainties 
  sigy_abs = 0.1 # abs. independent errors on y
  sigy_rel = 0.05 # relative error on y
  syabscor=0.1  #  an absolute, correlated error on y
  sigx_abs = 0.1 # absolute error on x
  sigx_rel = 0.  # relative error on x
  sxrelcor=0.02  #  a relative, correlated error on x 

# generate the data
  xmin =  1.
  xmax =  10.
  xdata=np.arange(xmin, xmax+1. ,1.)
  nd=len(xdata)
  mpars=[0.3, -1.5, 0.5]
  np.random.seed(31415)
  xt, yt, ydata = generateXYdata(xdata, model, sigx_abs, sigy_abs, 
    srely=sigy_rel, xrelcor=sxrelcor, yabscor=syabscor, mpar=mpars )

# fit with kafe2
  par, pare, cor, chi2 = k2Fit(model,
    xdata, ydata,        # data and uncertainties
    sx=sigx_abs,         # indep. x abs.
    sy=sigy_abs,         # indep. y abs.
    srelx=None,          # indep. x corr.
    srely=sigy_rel,      # indep. y corr.
    xabscor=None,        # correlated x abs.
    xrelcor=sxrelcor,    # correlated x rel.
    yabscor=syabscor,    # correlated y abs.
    yrelcor=None,        # correlated y rel.
    ref_to_model=True,   # reference of rel. uncert. to model
    p0=(0.5,-1.,1.),     # initial guess for parameter values
    plot=True,           # generate result plot (see options below)  
    fit_info=True,       # suppress fit results in figure if False
    plot_band=True,      # suppress model uncertainty-band if False
    plot_cor=False,      # plot profiles and contours
    showplots=True,     # plots on screen, plt.show() in user code if False
    axis_labels=['x-data','random y'], # set nice names
    data_legend = 'random data',       # legend entry for data points
    model_name = r'f\,',               # name for model
    model_expression = r'{a}\,{x}^2 + {b}\,{x} + {c}',  # model fuction
    model_legend = 'quadratic model',  # legend entry for model line
    model_band = None                  # name for model uncertainty band
    )                      
# setting any of the above names to None will remove the entry from the legend,
#  if not specified, default is used  

  print('*==* data set')
  print('  x = ', xdata)
  print('  y = ', ydata)
  np.set_printoptions(precision=3)
  print('  sx = ', np.sqrt(sigx_abs**2 + (sigx_rel*xdata)**2) )
  print('  sy = ', np.sqrt(sigy_abs**2 + (sigy_rel*model(xdata, *par))**2) )

  print('*==* fit result:')
  print("  -> chi2:         %.3g"%chi2)
  np.set_printoptions(precision=3)
  print("  -> parameters:   ", par)
  np.set_printoptions(precision=2)
  print("  -> uncertainties:", pare) 
  print("  -> correlation matrix: \n", cor) 

