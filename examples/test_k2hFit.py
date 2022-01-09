def k2hFit(fitf, data, bin_edges, 
         p0 = None, constraints = None,
         fixPars=None, limits=None,
         use_GaussApprox = False,
         plot = True, plot_cor = False,
         showplots = True, plot_band=True, plot_residual=False,
         quiet = True,
         axis_labels=['x', 'counts/bin = f(x, *par)'],
         data_legend = 'Histogram Data',    
         model_legend = 'Model',
         model_expression=None,
         model_name=None, 
         model_band = r'$\pm 1 \sigma$',           
         fit_info=True, asym_parerrs=True):

  
  """Wrapper function to fit a density distribution f(x, \*par) 
  to binned data (histogram) with class mnFit 
  
  The cost function is two times the negative log-likelihood of the 
  Poisson  distribution, or - optionally - of the Gaussian approximation.

  Uncertainties are determined from the model values in order to avoid biases 
  and to take account of empty bins of an histogram. 

  Args:
    * fitf: model function to fit, arguments (float:x, float: \*args)
    * data: the data to be histogrammed  
    * bin_edges: bin edges 

    fit options

      * p0: array-like, initial guess of parameters
      * constraints: (nested) list(s) [name or id, value, error] 
      * limits: (nested) list(s) [name or id, min, max]
      * use_GaussApprox: Gaussian approximation instead of Poisson 

    output options

      * plot: show data and model if True
      * plot_cor: show profile likelihoods and confidence contours
      * plot_band: plot uncertainty band around model function
      * plot_residual: also plot residuals w.r.t. model
      * showplots: show plots on screen
      * quiet: suppress printout
      * axis_labes: list of tow strings, axis labels
      * data_legend: legend entry for data
      * model_legend: legend entry for model 
      * plot: flag to switch off graphical output
      * axis_labels: list of strings, axis labels x and y
      * model_name: latex name for model function
      * model_expression: latex expression for model function
      * model_band: legend entry for model uncertainty band
      * fit_info: controls display of fit results on figure
      * asym_parerrs: show (asymmetric) errors from profile-likelihood scan

  Returns:
    * list: parameter names
    * np-array of float: parameter values
    * np-array of float: negative and positive parameter errors
    * np-array: cor   correlation matrix 
    * float: goodness-of-fit (equiv. chi2 for large number of entries/bin)

  """
  
  # for fit with kafe2
  from kafe2 import HistContainer, Fit, Plot, ContoursProfiler
  from kafe2.fit.histogram import HistCostFunction_NegLogLikelihood
  
  # create a data container from input
  nbins = len(bin_edges)-1
  bin_range = (bin_edges[0], bin_edges[-1])
  hdat = HistContainer(nbins, bin_range, bin_edges=bin_edges, fill_data=data)

  # set up fit object
  if use_GaussApprox:
    print('Gauss Approx. for histogram data not yet implemented - exiting!')
    exit(1)
     ## hfit = Fit(hdat, fitf,
     ##            cost_function=CostFunction_GaussApproximation)
  else:   
     hfit = Fit(hdat, fitf,
                cost_function=HistCostFunction_NegLogLikelihood(
                  data_point_distribution='poisson') )
  # text for labeling       
  hfit.assign_model_function_latex_name(model_name)
  hfit.assign_model_function_latex_expression(model_expression)
  hfit.model_label = model_legend

  # - provide text for labeling ...      
  hdat.label = data_legend
  hdat.axis_labels = axis_labels

  # initialize and run fit
  if p0 is not None: hfit.set_all_parameter_values(p0)

  if constraints is not None:
    if not (isinstance(constraints[0], tuple) or isinstance(constraints[0], list)):
      constraints = (constraints,)
    for c in constraints:
      hfit.add_parameter_constraint(*c)

  if limits is not None:
    if isinstance(limits[1], list):
      for l in limits:
        hfit.limit_parameter(l[0], l[1], l[2])          
    else: 
      hfit.limit_parameter(limits[0], limits[1], limits[2])          

  hfit.do_fit()                        

  # harvest results
  #  par, perr, cov, chi2 = fit.get_results() # for kafe vers. > 1.1.0
  parn = np.array(hfit.parameter_names) 
  parv = np.array(hfit.parameter_values) 
  pare = np.array(hfit.parameter_errors)
  cor = np.array(hfit.parameter_cor_mat)
  gof = hfit.goodness_of_fit
  if asym_parerrs:
    parae = np.array(hfit.asymmetric_parameter_errors)
  else:
    parae = np.array(list(zip(-pare, pare)))

  if not quiet:
    hfit.report(asymmetric_parameter_errors=True)

  if plot:
   # plot data, uncertainties, model line and model uncertainties
    kplot=Plot(hfit)
    # set some 'nice' options
    kplot.customize('data', 'marker', ['o'])
    kplot.customize('data', 'markersize', [6])
    kplot.customize('data', 'color', ['darkblue'])
## the following not (yet) defined for kafe2 Histogram Fit     
##    kplot.customize('model_line', 'color', ['darkorange'])
##    kplot.customize('model_line', 'linestyle', ['--'])
##    if not plot_band:
##      kplot.customize('model_error_band', 'hide', [True])
##    else:
##      kplot.customize('model_error_band', 'color', ['green'])
##      kplot.customize('model_error_band', 'label', [model_band])
##      kplot.customize('model_error_band', 'alpha', [0.1])     

    # plot with defined options
    kplot.plot(fit_info=fit_info, residual=plot_residual,
               asymmetric_parameter_errors=True)

    if plot_cor:
      cpf = ContoursProfiler(hfit)
      cpf.plot_profiles_contours_matrix() # plot profile likelihood and contours

    if showplots: plt.show()    
      
  return parv, parae, cor, gof

import numpy as np, matplotlib.pyplot as plt
##from PhyPraKit.phyFit import hFit
from PhyPraKit import hFit

if __name__ == "__main__": # --------------------------------------  
  #
  # Example of a histogram fit
  #

  # define the model function to fit
  def model(x, mu = 6.0, sigma = 0.5, s = 0.3):
    '''pdf of a Gaussian signal on top of flat background
    '''
    normal = np.exp(-0.5*((x-mu)/sigma)**2)/np.sqrt(2.*np.pi*sigma**2)
    flat = 1./(max-min) 
    return s * normal + (1-s) * flat

  #
  # generate Histogram Data
  #
  # parameters of data sample, signal and background parameters
  N = 100  # number of entries
  min=0.   # range of data, mimimum
  max= 10. # maximum
  s = 0.25 # signal fraction 
  pos = 6.66 # signal position
  width = 0.33 # signal width
  
  # fix random generator seed 
  np.random.seed(314159)  # initialize random generator

  def generate_data(N, min, max, p, w, s):
    '''generate a random dataset: 
       Gaussian signal at position p with width w and signal fraction s 
       on top of a flat background between min and max
     '''
     # signal sample
    data_s = np.random.normal(loc=pos, scale=width, size=int(s*N) )
     # background sample
    data_b = np.random.uniform(low=min, high=max, size=int((1-s)*N) )
    return np.concatenate( (data_s, data_b) )

  # generate a data sample ...
  SplusB_data = generate_data(N, min, max, pos, width, s)  
# ... and create the histogram
  bc, be = np.histogram(SplusB_data, bins=40)

#  
# ---  perform fit  
#
  parv, pare, cor, gof = k2hFit(model, SplusB_data, 
      be,             # bin bin edges
      p0=None,        # initial guess for parameter values 
   #  constraints=['name', val ,err ],   # constraints within errors
      limits=('s', 0., None),  #limits
      ## use_GaussApprox=True,   # Gaussian approximation does not work yet ...
      ## fit_density = True,      # fit density (not yet ...)
      plot=True,           # plot data and model
      plot_band=True,      # plot model confidence-band
      plot_residual=True,  # show residual w.r.t. model
      plot_cor=True,      # plot profiles likelihood and contours
      quiet=False,         # suppress informative printout
      axis_labels=['x', 'entries / bin   \  f(x, *par)'], 
      data_legend = 'pseudo-data',    
      model_legend = 'model',
      model_name = r'N\,',       # name for model
      model_expression = r'N({mu},{sigma},{s})',  # model fuction
      model_band = None          # name for model uncertainty band         
  )

# Print results to illustrate how to use output
  print('\n*==* Results of kafe2 Histgoram Fit:')  
  print("  -> gof:         %.3g"%gof)
  np.set_printoptions(precision=3)
  print("  -> parameters:   ", parv)
  np.set_printoptions(precision=2)
  print("  -> uncertainties:\n", pare) 
  print("  -> correlation matrix: \n", cor) 
