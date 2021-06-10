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

# --- end helper functions ----
import sys
import numpy as np, matplotlib.pyplot as plt
from PhyPraKit import generateXYdata, xyFit, k2Fit, plotCorrelations
from PhyPraKit.phyFit import get_functionSignature

from scipy import stats

if __name__ == "__main__": # --------------------------------------  
  #
  # Example of fitting with PhyPraKit functions xyFit, k2Fit and odFit
  #
  # Fits are run in a loop and 1d and 2d distribution of results plotted.
  
  # define the model function to fit
  def exp_model(x, A=1., x0=1.):
    return A*np.exp(-x/x0)

  # -- another model function
  def poly2_model(x, a=0.75, b=1., c=1.):
    return a*x**2 + b*x + c

  #set parameter of toy MC
  if len(sys.argv)==2:
    Nexp=int(sys.argv[1])
  else:
    Nexp = 1000
  model = poly2_model
#  model = exp_model
  
 # set error components (!!! odFit only supports non-zero sabsy and sabsx)
  # independent uncertainties
  sabsy = 0.07 # 0.07 
  sabsx = 0.05 # 0.05
  
 # the following are only supported by kafe2 and phyFit
  # independent relative uncertainties
  srely = 0.05 # 5% of model value
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
  mpardict = get_functionSignature(model)[1] # keyword arguments of model
  true_vals = np.asarray(list(mpardict.values()))
  npar=len(mpardict)
  parnams = list(mpardict.keys())
  sigy = np.sqrt(sabsy * sabsy + (srely*model(data_x, **mpardict))**2)
  sigx = np.sqrt(sabsx * sabsx + (srelx * data_x)**2)

  # initialize arrays for statistical analysis in loop
  nfail = 0                      # failed fits
  biases = [[] for i in range(npar)]  # bias
  c2prb = []                     # chi^2 probatility
  N_coverage = npar*[0]          # counters for coverage

  # uncertainties 
  kw_uncertainties = {
      'sx': sabsx,            # indep x
      'sy': sabsy,            # indel y
      'srelx': srelx,         # indep. rel. x
      'srely': srely,         # indep. rel. y
      'xabscor': cabsx,       # correlated x
      'xrelcor': crelx,       # correlated rel. x
      'yabscor': cabsy,       # correlated y
      'yrelcor': crely       # correlated rel. y
     }
  # fit options
  kw_fitoptions= {
      'ref_to_model': True,   # reference of rel. uncert. to model
      'use_negLogL' : True    # use full 2 neg log L, chi2 if False    
      }
  
  # set the fitting function  
  #  theFit = odFit    
  #  theFit = k2Fit    
  theFit = xyFit

  def MC_loop():
    global fitResults, nfail, biases, c2prb, N_coverage

    # run MC loop
    for iexp in range(Nexp):
      # generate pseudo data
      np.random.seed(314159+(iexp*2718281)%100000) # initialize random generator
   
      xt, yt, data_y = generateXYdata(data_x, model, sigx, sigy,
                                      xabscor=cabsx,
                                      xrelcor=crelx,
                                      yabscor=cabsy,
                                      yrelcor=crely,
                                      mpar=true_vals )
    # perform fit to pseudo data 
      try:
        if iexp<=0:   # show data and best-fit model
          plot=True
        else:
          plot=False
        fitResults = theFit(model,
        data_x, data_y,      # data x and y coordinates
        **kw_uncertainties,
        **kw_fitoptions,                                              
##        p0=(1., 0.5),        # initial guess for parameter values 
##        constraints=['A', 1., 0.03], # constraints within errors
##        fixPars= [1],         # fix parameters in fit 
        quiet=True,          # suppress informative printout
        plot=plot,           # plot data and model
#        plot_band=False,      # plot model confidence-band
        plot_cor=False,      # plot profiles likelihood and contours
        showplots=False,     # call plt.show() in user code if False
        axis_labels=['x', 'y   \  f(x, *par)'], 
        data_legend = 'pseudo-data',    
        model_legend = 'model'
        )
        
        pnams, pvals, perrs, cor, chi2 = fitResults
        # Print results to illustrate how to use output
        np.set_printoptions(precision=6)
        print('\n*==*  Fit {:d} Result:'.format(iexp))
        print(f" chi2: {chi2:.3g}")
        print(f" parameter names:  ", pnams)
        print(f" parameter values:  ", pvals)
        np.set_printoptions(precision=3)
        print(f" parameter errors: ", perrs)
        np.set_printoptions(precision=3)
        print(f" correlations : \n", cor)
        np.set_printoptions(precision=8) # default output precision

        #  calculate chi2 probability
        chi2prb = 1.- stats.chi2.cdf(chi2, nd-len(pvals))
        c2prb.append(chi2prb)
        # analyze bias and coverage
        for i in range(npar):
          biases[i].append( pvals[i] - true_vals[i] )  # bias
          # get parameter confidence interval (CI)
          pmn = pvals[i] + perrs[i,0]
          pmx = pvals[i] + perrs[i,1]
          # coverage: count how often true value is in CI
          if (true_vals[i] >= pmn and true_vals[i]<=pmx): N_coverage[i] +=1
                
        if plot:
          plt.suptitle('fit {:d} result'.format(iexp))
          ## plt.show() # blocks at this stage until figure deleted
     
      except Exception as e:
        nfail +=1
        print('!!! fit failed ', nfail)
        print(e)
  # - end MC loop 

  def print_results():
    # print overview of fit results
    global fitResults, nfail, biases, c2prb, N_coverage
    pnams, pvals, perrs, cor, chi2 = fitResults
    # analyze results
    # - convert to numpy arrays
    for i in range(npar):
      biases[i] = np.array(biases[i])
    c2prb = np.array(c2prb)

    # printout
    N_succ = Nexp - nfail 
    print('\n\n*==* ', N_succ, ' successful fits done:')
    print(' * parameter names:')
    for i in range(npar):
      print('   {:d}: {:s}'.format(i, pnams[i])) 
    print(' * biases:')
    for i in range(npar):
    #  bias = deviation of parameters from their true values
      b = biases[i].mean()
      e = biases[i].std()/np.sqrt(Nexp)
      print('   {:d}: {:.3g} +\- {:.2g}'.format(i, b, e)) 
    print(' * coverage:')
    for i in range(npar):
    #  coverage: fraction of true val in confidence interval
      p_coverage = N_coverage[i]/(N_succ)*100/0.682689492
      print('   {:d}: {:.3g}%'.format(i, p_coverage))

  def plot_correlations():
    # show parameter distributions and correlations 
    global fitResults, nfail, biases, c2prb, N_coverage
    pnams, pvals, perrs, cor, chi2 = fitResults

    names = [r'$\Delta$'+pnams[i] for i in range(len(pnams))]
    plotCorrelations(biases, names)
    plt.suptitle("Biases and Correlations", size='xx-large')
    ax = plt.gca()
    ax.text(0.1, 0.45,                
                '$\\Delta$: fit - true \n \n' +
                '$\\mu$: mean \n' +
                '$\\sigma$: standard deviation \n' +
                '$\\sigma_\\mu$: error on mean \n \n' +
                '$\\rho$: correlation coefficient',
                transform=ax.transAxes)

  def analyse_chi2():
    # analyse goodness of fit variable
    global fitResults, nfail, biases, c2prb, N_coverage
    # analyse chi2 probability
    figc2prb = plt.figure(figsize=(7.5, 5.))
    ax = figc2prb.add_subplot(1,1,1)
    nbins = int(min(50, Nexp/20))
    binc, bine, _ = ax.hist(c2prb, bins=nbins, ec="grey")
    plt.xlabel('chi2 probability')
    plt.ylabel('entries/bin')
    # - check compatibility with flat distribution
    mean = (Nexp-nfail)/nbins
    chi2flat= np.sum( (binc - mean)**2 / mean )
    prb = 1.- stats.chi2.cdf(chi2flat, nbins)
    print('compatibility of chi2prb with flat distribution: {:f}%'.format(prb*100))

  #
  # -- execute all of the above --
  # 
  MC_loop()
  print_results()
  plot_correlations()
  analyse_chi2()
  
  plt.show()
