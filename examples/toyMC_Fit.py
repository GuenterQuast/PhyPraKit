#! /usr/bin/env python3
"""toyMC_Fit.py
   run a large number of fits on toyMC data
   to check for biases and chi2-probability distribution

   This rather complete example uses eight different kinds of uncertainties, 
   namely independent and correlated, absolute and relative ones  
   in the x and y directions. 
   
.. moduleauthor:: Guenter Quast <g.quast@kit.edu>

"""

# --- Helper functions
import inspect
def get_signature(f):
  # get arguments and keyword arguments passed to a function
  pars = inspect.signature(f).parameters
  args = []
  kwargs = {}
  for p in pars.values():
    if p.default is p.empty:
      args.append(p.name)
    else:
      kwargs[p.name]=p.default
  return args, kwargs

# --- end helper functions ----

import numpy as np, matplotlib.pyplot as plt
from PhyPraKit import generateXYdata, mFit, k2Fit

from scipy import stats

if __name__ == "__main__": # --------------------------------------  
  #
  # Example of fitting with PhyPraKit functions mFit, k2Fit and odFit
  #

  # define the model function to fit
  def exp_model(x, A=1., x0=1.):
    return A*np.exp(-x/x0)

  # -- another model function
  def poly2_model(x, a=1.5, b=1., c=1.):
    return a*x**2 + b*x + c

  #set parameter of toy MC
  Nexp = 1000
  model = poly2_model
#  model = exp_model
  
 # set error components (!!! odFit only supports non-zero sabsy and sabsx)
  # indepentent uncertainties
  sabsy = 0.07 # 0.07 
  sabsx = 0.05 # 0.05
  
 # the following are only supported by kafe2 and iminuitFit
  # indepentent relative uncertainties
  srely = 0.03 # 5% of model value
  srelx = 0.04 # 4%
  # correlated uncertainties y and x direction 
  cabsy = 0.04 # 0.04
  crely = 0.03 # 3% of model value
  cabsx = 0.03 # 0.03
  crelx = 0.02 # 2%
  
 # parameters of pseudo data
  nd=120   # number of data points
  xmin=0.  # x-range
  xmax=2.5
  data_x = np.linspace(xmin, xmax, nd)       # x of data points
  mpardict = get_signature(model)[1] # keyword arguments of model
  npar=len(mpardict)
  sigy = np.sqrt(sabsy * sabsy + (srely*model(data_x, **mpardict))**2)
  sigx = np.sqrt(sabsx * sabsx + (srelx * data_x)**2)

# set the fitting function  
#  theFit = odFit    
#  theFit = k2Fit    
  theFit = mFit    

  # initialize arrays for statistical analysis in loop
  d = [[] for i in range(npar)]
  c2prb = []  
  nfail = 0
  for i in range(Nexp):  
    # keep random numbers under control
    np.random.seed(314159+i)     # initialize random generator
    
    xt, yt, data_y = generateXYdata(data_x, model, sigx, sigy,
                                      xabscor=cabsx,
                                      xrelcor=crelx,
                                      yabscor=cabsy,
                                      yrelcor=crely,
                                      mpar=mpardict.values() )

# perform fit to data 
    try:
      if i<=0:   # show data and best-fit model
        plot=True
      else:
        plot=False
      parvals, parerrs, cor, chi2 = theFit(model,
      data_x, data_y,      # data x and y coordinates
      sx=sabsx,            # indep x
      sy=sabsy,            # indel y
      srelx=srelx,         # indep. rel. x
      srely=srely,         # indep. rel. y
      xabscor=cabsx,       # correlated x
      xrelcor=crelx,       # correlated rel. x
      yabscor=cabsy,       # correlated y
      yrelcor=crely,       # correlated rel. y
      ref_to_model=True,   # reference of rel. uncert. to model
##      p0=(1., 0.5),        # initial guess for parameter values 
#       constraints=['A', 1., 0.03], # constraints within errors
#      use_negLogL=True,    # full -2log(L) if parameter dep. uncertainties
      plot=plot,           # plot data and model
#      plot_band=True,      # plot model confidence-band
      plot_cor=False,      # plot profiles likelihood and contours
      showplots=False,     # call plt.show() in user code if False
      quiet=True,          # suppress informative printout
      axis_labels=['x', 'y   \  f(x, *par)'], 
      data_legend = 'random data',    
      model_legend = 'exponential model'
      )

  # Print results to illustrate how to use output
      print('\n*==* ', i, ' Fit Result:')
      print(f" chi2: {chi2:.3g}")
      print(f" parameter values: ", parvals)
      print(f" parameter errors: ", parerrs)
      print(f" correlations : \n", cor)
      chi2prb = 1.- stats.chi2.cdf(chi2, nd-len(parvals))
    # analyze
      for i in range(npar):
        d[i].append(parvals[i]-list(mpardict.values())[i])
      c2prb.append(chi2prb)

      if plot: plt.show()
    
    except Exception as e:
      nfail +=1
      print('!!! fit failed ', nfail)
      print(e)

  # - end loop over MC experiments    

  for i in range(npar):
    d[i] = np.array(d[i])
    c2prb = np.array(c2prb)
  
  # print deviation of parameters from their true values
  print('\n\n*====* ', Nexp - nfail, ' successful fits done:')
  for i in range(npar):
    print('   delta par',i,' ', d[i].mean()) 
    print('        +/-     ', d[i].std()/np.sqrt(Nexp)) 

  # analyse chi2 probability
  nbins= int(min(50, Nexp/20))
  binc, bine, _ = plt.hist(c2prb, bins=nbins, ec="grey")
  mean = (Nexp-nfail)/nbins
  chi2flat= np.sum( (binc - mean)**2 / mean )
  prb = 1.- stats.chi2.cdf(chi2flat, nbins)
  print(f'compatibility of chi2prb with flat distribution {prb*100:f}%')  

  plt.xlabel('chi2 probability')
  plt.ylabel('entries/bin')
  plt.show()
