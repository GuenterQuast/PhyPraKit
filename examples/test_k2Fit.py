#! /usr/bin/env python3
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

# import kafe2 # must be imported first to properly set matplotlib backend
from PhyPraKit import generateXYdata, k2Fit
import numpy as np, matplotlib.pyplot as plt

# -- the model function
#
# Example of an application
# 
# model function to fit
def expModel(x, A=1., x0=1.):
  return A*np.exp(-x/x0)

if __name__ == "__main__": # --------------------------------------  

  model=expModel
  mpardict = {'A':1., 'x0':1.}  # model parameters

# set error components 
  sabsy = 0.07
  srely = 0.05 # 5% of model value
  cabsy = 0.04
  crely = 0.03 # 3% of model value
  sabsx = 0.05
  srelx = 0.04 # 4% 
  cabsx = 0.03 
  crelx = 0.02 # 2% 

# ---> generate pseudo data
  np.random.seed(314159)      # initialize random generator
  nd=14
  xmin=0.
  xmax=2.6
  data_x = np.linspace(xmin, xmax, nd)       # x of data points
  sigy = np.sqrt(sabsy * sabsy + (srely*model(data_x, **mpardict))**2)
  sigx = np.sqrt(sabsx * sabsx + (srelx * data_x)**2)
  xt, yt, data_y = generateXYdata(data_x, model, sigx, sigy,
                                      xabscor=cabsx,
                                      xrelcor=crelx,
                                      yabscor=cabsy,
                                      yrelcor=crely,
                                      mpar=mpardict.values() )
# <---- end generate pseudo data

# fit data with kafe2
  par, pare, cor, chi2 = k2Fit(model,
    data_x, data_y,
    sabsx,
    sabsy,                                    # data and uncertaintites
    srelx=srelx,
    srely=srely,
    xabscor=cabsx,
    xrelcor=crelx,
    yabscor=cabsy,                            # correlated uncertainties
    yrelcor=crely,
    ref_to_model=True,
    p0=(1., .5),                              # initial guess and range
    quiet=True,
    plot=True,                                # show plot, options below  
    fit_info=True,                             # fit results in figure
 #   plot_band=False,
    plot_cor=True,
    axis_labels=['x-data','random y'],        # nice names
    data_legend = 'random data',              # legend entry for data points
    model_name = r'f\,',                 # name for model
    model_expression = r'{A}\,\exp({x}/{x0})',   # model fuction
    model_legend = 'exponential model',       # legend entry for model line
    #model_band = None,                       # name for model uncertainty band
   )
# setting any of the above names to None will remove the entry from the legend,
#  if not specified, use default  

  print('*==* fit result:')
  print("  -> chi2:         %.3g"%chi2)
  np.set_printoptions(precision=3)
  print("  -> parameters:   ", par)
  np.set_printoptions(precision=2)
  print("  -> uncertainties:", pare) 
  print("  -> correlation matrix: \n", cor) 

