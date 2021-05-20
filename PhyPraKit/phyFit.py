"""package phyFit.py
  
  Physics Fitting with `iminiut` [https://iminuit.readthedocs.ios/en/stable/]

  Author: Guenter Quast, initial version Jan. 2021

  Requirements: 
   - Python >= 3.6
   - iminuit vers. > 2.0
   - scipy > 1.5.0
   - matplotlib > 3

  The class `mnFit.py` uses the optimization and uncertainty-estimation
  package `iminuit` for fitting a parameter-dependent model f(x, \*par) 
  to data points (x, y) or a probability density function to binned 
  histogram data or to unbinned data. Parameter estimation is based on 
  the Maximum-Likelihood method in the first two cases, or on a user-defined
  likelihood function in the latter case. Classical least-square methods 
  are optionally available for comparison with other packages. 

  A unique feature of the package ist the support of different kinds 
  of uncertainties for x-y data, namely independent and/or correlated 
  absolute and/or relative uncertainties in the x and/or y directions. 
  Parameter estimation for density distributions is based on the shifted 
  Poission distibution, Poisson(x-loc, lambda), of the number of entries 
  in each bin of a histogram. 

  Parameter constraints, i.e. external knowledge of parameters within
  Gaussian uncertainties, and limits on parameters in order to avoid 
  problematic regions in parameter space during the minimization process, 
  are also supported by mnFit.

  Method:
    Uncertainties that depend on model parameters are treated by dynamically 
    updating the cost function during the fitting process wit `iminuit`.
    Data points with relative errors can thus be referred to the model 
    instead of the data. The derivative of the model function w.r.t. x is 
    used to project the covariance matrix of x-uncertainties on the y-axis. 

  Example functions mFit() and hFit() illustrate how to control 
  the  interface of `mnFit` for x-y and histogram fits, and a 
  short script is provided to perform fits on sample data.  
  A brief example how to implement fit of a prbability density to
  a set of (unbinned) data is also provided. 

  The implementation of the fitting procedure in this package is 
  - intenitionally - rather minimalistic, and it is intended to 
  illustrate the principle of an advanced usage of `iminuit`. It is 
  also meant to stimulate own applications of special, user-defined cost 
  functions.

  The main features of this package are:
    - provisioning of cost functions for x-y and binned histogram fits 
    - implementation of the least-squares method for correlated Gaussian errors
    - support for correlated x-uncertainties by projection on the y-axis
    - support of relative errors with reference to the model values 
    - evaluation of profile likelihoods to determine asymetric uncertainties
    - plotting of profile likeliood and confidence contours

  The **cost function** that is optimized for x-y fits basically is a 
  least-squares one, which is extended if parameter-dependent uncertainties 
  are present. In the latter case, the logarithm of the determinant of the 
  covariance matrix is added to the least-squares cost function, so that it 
  corresponds to twice the negative log-likelihood of a multivariate Gaussian 
  distribution. Fits to bistogram data rely on the negative log-likelihood
  of the Poisson distribution, which is generalised to support fractional
  observed values, which may occur if corrections to the observed bin counts 
  have to be applied. If there is a difference between the mean value and
  the variance of the number of entries in a bin due to corrections, 
  a "shifted Poisson distribution", Poiss(x-DeltaMu, lambda), is supported.

  Fully functional examples are provided by the functions `mFit()` and
  `hFit()` and the executable script below, which contains sample data, 
  executes the fitting procecure and collects the results. The script also
  contains only a few lines of code to perform a very minimalistic fit
  to unbinnde data with a user-defined likelihood function. 
  
.. moduleauthor:: Guenter Quast <g.quast@kit.edu>
"""

import sys
import numpy as np, matplotlib.pyplot as plt
from scipy import stats, linalg
from inspect import signature
from iminuit import __version__, Minuit
from scipy.special import loggamma


def mFit(fitf, x, y, sx = None, sy = None,
       srelx = None, srely = None, 
       xabscor = None, xrelcor = None,        
       yabscor = None, yrelcor = None,
       ref_to_model = True, 
       p0 = None, constraints = None, limits=None,
       use_negLogL=True, 
       plot = True, plot_cor = True,
       showplots = True, 
       plot_band=True, quiet = False,
       axis_labels=['x', 'y = f(x, *par)'],
       data_legend = 'data',    
       model_legend = 'model'):
  
  """Wrapper function to fit an arbitrary function fitf(x, \*par) 
  to data points (x, y) with independent and/or correlated absolute 
  and/or relative errors  on x- and/or y- values with class mnFit

  Correlated absolute and/or relative uncertainties of input data 
  are specified as floats (if all uncertainties are equal) or as 
  numpy-arrays of floats. The concept of independent or common 
  uncertainties of (groups) of data points is used construct the 
  full covariance matrix from different uncertainty components.
  Indepenent uncertainties enter only in the diagonal, while correlated 
  ones contribute to diagonal and off-diagonal elements of the covariance 
  matrix. Values of 0. may be specified for data points not affected by a 
  certrain type of uncertainty. E.g. the array [0., 0., 0.5., 0.5] specifies
  uncertainties only affecting the 3rd and 4th data points. Providing lists 
  of such arrays permits the construction of arbitrary covariance matrices 
  from independent and correlated uncertainties of (groups of) data points.

  Args:
    * fitf: model function to fit, arguments (float:x, float: \*args)
    * x:  np-array, independent data
    * y:  np-array, dependent data
    * sx: scalar or 1d or 2d np-array , uncertainties on x data
    * sy: scalar or 1d or 2d np-array , uncertainties on x data
    * srelx: scalar or np-array, relative uncertainties x
    * srely: scalar or np-array, relative uncertainties y
    * yabscor: scalar or np-array, absolute, correlated error(s) on y
    * yrelcor: scalar or np-array, relative, correlated error(s) on y
    * p0: array-like, initial guess of parameters
    * use_negLogL:  use full -2ln(L)  
    * constraints: (nested) list(s) [name or id, value, error] 
    * limits: (nested) list(s) [name or id, min, max] 
    * plot: show data and model if True
    * plot_cor: show profile liklihoods and conficence contours
    * plot_band: plot uncertainty band around model function
    * showplots: show plots on screen
    * quiet: suppress printout
    * list of str: axis labels
    * str: legend for data
    * str: legend for model 

  Returns:
    * np-array of float: parameter values
    * 2d np-array of float: parameter uncertaities [0]: neg. and [1]: pos. 
    * np-array: correlation matrix 
    * float: 2*negLog L, corresponding to \chi-square of fit a minimum
  """

  ## from .phyFit import mnFit #! already contained in this file

  # ... check if errors are provided ...
  if sy is None:
    sy = np.ones(len(y))
    print('\n!**! No y-errors given, all assumed to be 1.0\n',
          '-> consider scaling of parameter errors with sqrt(chi^2/Ndf)\n')
  
  # set up a fit object
  Fit = mnFit()

  # set some options
  Fit.setOptions(run_minos=True,
                 relative_refers_to_model=ref_to_model,
                 use_negLogL=use_negLogL,
                 quiet=quiet)

  # pass data and uncertainties to fit object
  Fit.init_data(x, y,
              ex = sx, ey = sy,
              erelx = srelx, erely = srely,
              cabsx = xabscor, crelx = xrelcor,
              cabsy = yabscor, crely = yrelcor)
   # pass model fuction, start parameter and possibe constraints
  Fit.init_fit(fitf, p0=p0,
               constraints=constraints,
               limits=limits)
   # perform the fit
  fitResult = Fit.do_fit()
  # print fit result(dictionary from migrad/minos(
  if not quiet:
    print("\nFit Result from migrad:")
    print(fitResult[0])
    if fitResult[1] is not None:
      print("\nResult of minos error analysis:")
      print(fitResult[1])
    
  # produce figure with data and model
  if plot:
    fig = Fit.plotModel(axis_labels=axis_labels,
                 data_legend=data_legend,
                 model_legend=model_legend,
                      plot_band=plot_band)

  # figure with visual representation of covariances
  #   profile likelihood scan and confidence contours
  if plot_cor:
    fig_cor = Fit.plotContours(figname="xyFit: Profiles and Contours")

  # show plots on screen
  if showplots and (plot or plot_cor):
    plt.show()

  # return
  #   numpy arrays with fit result: parameter values,
  #   negative and positive parameter uncertainties,
  #   correlation matrix
  #   gof
  return Fit.getResult()

def hFit(fitf, bin_contents, bin_edges, DeltaMu=None,
       p0 = None, constraints = None, limits=None,
       use_GaussApprox = False,
       fit_density = True,
       plot = True, plot_cor = True,
       showplots = True, plot_band=True,
       quiet = False,
       axis_labels=['x', 'counts/bin = f(x, *par)'],
       data_legend = 'Histogram Data',    
       model_legend = 'Model'):
  
  """Wrapper function to fit a density distribution f(x, \*par) 
  to binned data (histogram) with calss mnFit 
  
  The cost function is two times the negative log-likelihood of the Poission 
  distribution, or - optionally - of the Gaussian approximation.

  Uncertainties are determined from the model values in order to avoid biases and to
  to take account of empty bins of an histogram. The default behaviour is to fit a
  normalised density; optionally, it is also possible to fit the number of bin entries.

  Args:
    * fitf: model function to fit, arguments (float:x, float: \*args)
    * bin_contents:  
    * bin_edges: 
    * DeltaMu: shift mean (=mu) vs. variance (=lam), for Poisson: mu=lam
    * p0: array-like, initial guess of parameters
    * constraints: (nested) list(s) [name or id, value, error] 
    * limits: (nested) list(s) [name or id, min, max] 
    * GaussApprox: Gaussion approximation instead of Poisson 
    * density: fit density (not number of events)
    * plot: show data and model if True
    * plot_cor: show profile liklihoods and conficence contours
    * plot_band: plot uncertainty band around model function
    * showplots: show plots on screen
    * quiet: suppress printout
    * axis_labes: list of tow strings, axis labels
    * data_legend: legend entry for data
    * model_legend: legend entry for model 

  Returns:
    * np-array of float: parameter values
    * 2d np-array of float: parameter uncertaities [0]: neg. and [1]: pos. 
    * np-array: correlation matrix 
    * float: 2*negLog L, corresponding to \chi-square of fit a minimum
  """

  ## from .phyFit import mnFit #! already contained in this file

  # set up a fit object for histogram fits
  Fit = mnFit('hist')

  # set default options
  Fit.setOptions(run_minos = True,
                 use_GaussApprox = use_GaussApprox,
                 fit_density = fit_density,
                 quiet = quiet)

  # pass data and uncertainties to fit object
  Fit.init_data(bin_contents, bin_edges, DeltaMu)
   # pass model fuction, start parameter and possibe constraints
  Fit.init_fit(fitf, p0=p0,
               constraints=constraints,
               limits=limits)
  # perform the fit
  fitResult = Fit.do_fit()
  # print fit result (dictionary from migrad/minos(
  if not quiet:
    print("\nFit Result from migrad:")
    print(fitResult[0])
    if fitResult[1] is not None:
      print("\nResult of minos error analysis:")
      print(fitResult[1])
    
  # produce figure with data and model
  if plot:
    fig = Fit.plotModel(axis_labels=axis_labels,
                 data_legend=data_legend,
                 model_legend=model_legend,
                      plot_band=plot_band)

  # figure with visual representation of covariances
  #   profile likelihood scan and confidence contours
  if plot_cor:
    fig_cor = Fit.plotContours(figname="histFit: Profiles and Contours")

  # show plots on screen
  if showplots and (plot or plot_cor):
    plt.show()

  # return
  #   numpy arrays with fit result: parameter values,
  #   negative and positive parameter uncertainties,
  #   correlation matrix
  #   gof
  return Fit.getResult()

def userFit(cost, p0 = None, 
            constraints = None, limits=None,
            neg2logL = True,
            plot_cor = True,
            showplots = True, quiet = False):
  
  """Wrapper function to directly fit a user-defined cost funtion

  This is the simplest fit possibe with the class mnFit. A user-defined
  cost function is minimized and an estimation of the parameter uncertainties performed

  Args:
    * cost: user-defined cost funtion to be minimized; the uncertaintiy estimation
      releys this being a negative log-likelihood function ('nlL') 
    * p0: array-like, initial guess of parameters
    * constraints: (nested) list(s) [name or id, value, error] 
    * limits: (nested) list(s) [name or id, min, max] 
    * neg2logL: use 2 * nlL (corresponding to a least-squares-type cost)
    * plot_cor: plot likelihood profiles and confidence contours of parameters
    * showplots: show plots on screen (can also be done by calling script)
    * quiet: contrlos verbose output
    """

  uFit = mnFit("user")
  # set options
  uFit.setOptions(run_minos = True,
                 neg2logL = neg2logL,
                 quiet = quiet)
  # no internal data, directly initialze fit with the supplied cost function
  uFit.init_fit(cost, p0=p0,
                constraints=constraints,
                limits=limits)
  # perform the fit
  fitResult = uFit.do_fit()
  # print fit result (dictionary from migrad/minos(
  if not quiet:
    print("\nFit Result from migrad:")
    print(fitResult[0])
    if fitResult[1] is not None:
      print("\nResult of minos error analysis:")
      print(fitResult[1])
    
  # figure with visual representation of covariances
  #    profile likelihood scan and confidence contours
  if plot_cor:
    fig_cor = uFit.plotContours(figname="userFit: Profiles and Contours")

  # show plots on screen
  if showplots and (plot or plot_cor):
    plt.show()

  # return
  #   numpy arrays with fit result: parameter values,
  #   negative and positive parameter uncertainties,
  #   correlation matrix
  #   gof
  return uFit.getResult()

#
# --- classes and functions
#
class mnFit():
  """**Fit an arbitrary funtion f(x, *par) to data**  
  with independent and/or correlated absolute and/or relative uncertainties

  This implementation depends on and heavily uses features of 
  the minimizer and uncertainty-estimator **iminuit**.
   
  Public Data member

  - fit_type: 'xy' (default) or 'hist', controls type of fit 

  Public methods:

  - init_data():        generic wrapper for init_*Data() methods
  - init_fit():         generic wrapper for init_*Fit() methods
  - setOptions():       generic wrapper for set_*Options() methods
  - do_fit():           generic wrapper for do_*Fit() methods
  - plotModel():        plot model function and data
  - plotContours():     plot profile likelihoods and confidence contours 
  - getResult():        access to final fit results 
  - getFunctionError(): uncertainty of model at point(s) x for parameters p
  - plot_Profile():     plot profile Likelihood for parameter
  - plot_clContour():   plot confidence level coutour for pair of parameters  
  - plot_nsigContour(): plot n-sigma coutours for pair of parameters  

  Methods:

  - init_xyData():       initialze xy data and uncertainties
  - init_hData():        initialze histogram data and uncertainties
  - init_xyFit():        initialize xy fit: data, model and constraints
  - init_hFit():         initialize histogram fit: data, model and constraints
  - init_mnFit():        initialize histogram simple minuit fit
  - set_xyOptions():     set options for xy Fit
  - set_hOptions():      set options for histogram Fit
  - set_mnOptions():     set options for simple minuit fit with external cost function
  - do_xyFit():          perform xy fit
  - do_hFit():           perform histogram fit
  - do_mnFit():          simple minuit fit with external, user-defined cost function
 
  Data members:

  - ParameterNames:     names of parameters (as specified in model function)
  - GoF:                goodness-of-fit, i.e. chi2 at best-fit point
  - NDoF:               number of degrees of freedom
  - ParameterValues:    parameter values at best-fit point
  - MigradErrors:       symmetric uncertainties
  - CovarianceMatrix:   covariance matrix
  - CorrelationMatrix:  correlation matrix
  - OneSigInterval:     one-sigma (68% CL) ranges of parameer values 
 
  - covx:     covariance matrix of x-data
  - covy:     covariance matrix of y-data 
  - cov:      combined covariance matrix, including projected x-uncertainties

  Instances of (sub-)classes:

  - minuit.\*: methods and members of Minuit object 
  - data.\*:   methods and members of data sub-class, 
    generic for xyData or hData 
  - costf.\*:  methods and members of cost sub-class, generic for xLSQ or hCost
  """

  def __init__(self, fit_type='xy'):
    """
    Type of fit:

    - 'xy'   : fit model model y(f(x; par) to data 
    - 'hist' : fit densitiy to binned data (i.e. histogram) 
    - 'user': user-supplied cost-function (i.e. neg. log-likelihood)
    """

    if fit_type not in ['xy', 'hist', 'user']:
      sys.exit(
        '!**! mnFit: invalid fit type ', fit_type, '- exiting!') 
    self.fit_type = fit_type

    # set default of all options
    #
    #   no data or model provided yet
    self.xyData = None
    self.hData = None
    self.data = None
    self.costf = None
    # no fit done yet
    self.migradResult = None
    self.minosResult = None
    self.migrad_ok = False
    self.minos_ok = False
    # default options
    self.refModel=True
    self.run_minos = True
    self.quiet = True
    # for xy Fit
    self.use_negLogL = True
    # for histogram fit
    self.use_GaussApprox = False
    self.fit_density = True
    # for fit with external cost
    self.neg2logL = True
    self.ErrDef = 1.

  def init_data(self, *args, **kwargs):
    if self.fit_type == 'xy':
      self.init_xyData(*args, **kwargs)
    elif self.fit_type == 'hist':
      self.init_hData(*args, **kwargs)
    elif self.fit_type == 'user':
      print("!**! mnFit: not data object definded for fit_type 'user'" )
    else:
      print("!**! unknown type of Fit ", self.fit_type)
      sys.exit('mnFit Error: invalid fit type')
    
  def setOptions(self, *args, **kwargs):
    if self.fit_type == 'xy':
      self.set_xyOptions(*args, **kwargs)
    elif self.fit_type == 'hist':
      self.set_hOptions(*args, **kwargs)
    elif self.fit_type == 'user':
      self.set_mnOptions(*args, **kwargs)
    else:
      print("!**! unknown type of Fit ", self.fit_type)
      sys.exit('mnFit Error: invalid fit type')    

  def init_fit(self, *args, **kwargs):
    if self.fit_type == 'xy':
      self.init_xyFit(*args, **kwargs)
    elif self.fit_type == 'hist':
      self.init_hFit(*args, **kwargs)
    elif self.fit_type == 'user':
      self.init_mnFit(*args, **kwargs)
    else:
      print("!**! unknown type of Fit ", self.fit_type)
      sys.exit('mnFit Error: invalid fit type')

  def do_fit(self, *args, **kwargs):
    if self.fit_type == 'xy':
      return self.do_xyFit()
    elif self.fit_type == 'hist':
      return self.do_hFit(*args, **kwargs)
    elif self.fit_type == 'user':
      return self.do_mnFit(*args, **kwargs)
    else:
      print("!**! unknown type of Fit ", self.fit_type)
      sys.exit('mnFit Error: invalid fit type')    

  #
  # --- code for xy Fit
  #      

  def set_xyOptions(self,
              relative_refers_to_model=None,
              run_minos=None,
              use_negLogL=None,
              quiet=None):

    """Define options for xy fit

       Args:
        - rel. errors refer to model else data
        - run minos else don*t run minos
        - use full neg2logL
        - don*t provide printout else verbose printout 
    """
    if relative_refers_to_model is not None:
      self.refModel = relative_refers_to_model
    if run_minos is not None:   
      self.run_minos = run_minos
    if use_negLogL is not None:   
      self.use_negLogL = use_negLogL
    if quiet is not None:
      self.quiet = quiet
    
  def init_xyData(self,
                x, y,             
                ex=None, ey=1.,
                erelx=None, erely=None,
                cabsx=None, crelx=None,
                cabsy=None, crely=None):

    """initialize data object

    Args:
      -  x:       abscissa of data points ("x values")
      -  y:       ordinate of data points ("y values")
      -  ex:      independent uncertainties x
      -  ey:      independent uncertainties y
      -  erelx:   independent relative uncertainties x
      -  erely:   independent relative uncertainties y
      -  cabsx:   correlated abolute uncertainties x
      -  crelx:   correlated relative uncertainties x
      -  cabsy:   correlated absolute uncertainties y
      -  crely:   correlated relative uncertainties y
      -  quiet:   no informative printout if True
    """
    
    # create data object and pass all input arguments
    self.xyData = self.xyDataUncertainties(self, x, y, ex, ey,
                    erelx, erely, cabsx, crelx, cabsy, crely,
                    quiet=self.quiet)
    self.data = self.xyData
    # set flags for steering of fit process in do_fit()
    self.iterateFit = self.xyData.has_xErrors or(
         self.xyData.has_rel_yErrors and self.refModel)

  def init_xyFit(self, model, p0=None,
               constraints=None,
               limits=None):
    """initialize fit object

    Args:
      - model: model function f(x; \*par)
      - p0: np-array of floats, initial parameter values 
      - constraints: (nested) list(s): [parameter name, value, uncertainty] 
        or [parameter index, value, uncertainty]
      - limits: (nested) list(s): [parameter name, min, max] 
        or [parameter index, min, max]
    """

    # get parameters of model function to set start values for fit
    args, model_kwargs = self.get_functionSignature(model)
    self.pnams = list.copy(list(model_kwargs.keys()))
    if p0 is not None:
      for i, pnam in enumerate(self.pnams):
        model_kwargs[pnam] = p0[i]    

    # create cost function
    self.costf = self.xLSQ(self,
                           self.xyData, model,
                           use_neg2logL= self.use_negLogL,
                           quiet=self.quiet)
    if limits is not None:
      self.setLimits(limits)

    if constraints is not None:
      self.costf.setConstraints(constraints)

    # create Minuit object
    if __version__ < '2':
      if self.quiet:
        print_level=0
      else:
        print_level=1
      if limits is not None:
        for i, pnam in enumerate(self.pnams):
          model_kwargs['limit_' + pnam] = self.limits[i]
          
      self.minuit = Minuit(self.costf, 
                           errordef=self.ErrDef,
                           print_level=print_level,
                           **model_kwargs )
    else:
      self.minuit = Minuit(self.costf, **model_kwargs)  
      self.minuit.errordef = self.ErrDef
      if self.quiet:
        self.minuit.print_level = 0
      if limits is not None:
        self.minuit.limits = self.limits       
      
  def setLimits(self, limits):
    """store parameter limits

    format: nested list(s) of type 
    [parameter name, min, max] or
    [parameter index, min, max]
    """

    # get parameter names (from cost function)
    self.limits=[ [None, None]] * len(self.costf.pnams)
    if isinstance(limits[1], list):
      for l in limits:
        if type(l[0])==type(' '):
          p_id = self.costf.pnam2id[l[0]]
        else:
          p_id = l[0]
        self.limits[p_id] = [l[1], l[2]]          
    else:
      if type(limits[0])==type(' '):
        p_id = self.costf.pnam2id[limits[0]]
      else:
        p_id = limits[0]
      self.limits[p_id]=[limits[1], limits[2]]          
      
  def do_xyFit(self):
    """perform all necessary steps of fit sequence
    """
    if self.xyData is None:
      print(' !!! mnFit: no data object defined - call init_data()')
      sys.exit('mnFit Error: no data object')
    if self.costf is None:
      print(' !!! mnFit: no fit object defined - call init_fit()')
      sys.exit('mnFit Error: no fit object')
    
    # summarize options
    if not self.quiet:
      print( '*==* mnFit starting (pre-)fit')
      print( '  Options:')
      if self.run_minos is not None:
        print( '     - performing minos profile likelihood scan')
      if self.refModel is not None:
        print( '     - relative uncertainties refer to model ')
      if self.iterateFit and self.use_negLogL is not None:
        print( '     - using negative log-likelihood')
      
    # perform initial fit
    try:
      self.migradResult = self.minuit.migrad()  # find minimum of cost function
      self.migrad_ok = True
    except Exception as e:
      self.migrad_ok = False
      print('*==* !!! fit with migrad failed')
      print(e)
      exit(1)

    # possibly, need to iterate
    if self.iterateFit:
      if not self.quiet:
        print( '*==* mnFit iterating',
               'to account for parameter-dependent uncertainties')
      # enable dynamic calculation of covariance matrix
      self.xyData._init_dynamicErrors(self.refModel, self.costf.model)

      # fit with dynamic recalculation of covariance matrix
      try:
        self.migradResult = self.minuit.migrad()
        self.migrad_ok = True
      except Exception as e:
        self.migrad_ok = False
        print('*==* !!! iteration of fit with migrad failed')
        print(e)
        exit(1)

    # run profile likelihood scan to check for asymmetric errors
    if self.run_minos:
      if not self.quiet:
        print( '*==* mnFit starting minos scan')
      try:  
        self.minosResult = self.minuit.minos()
        self.minos_ok = True
      except Exception as e:
        self.minos_ok = False
        if not self.quiet:
          print( '*==* mnFit: !!! minos failed \n', e)    
    self._storeResult()
    return self.migradResult, self.minosResult
  
  class xyDataUncertainties:
    """
    Handle data and uncertainties, 
    build covariance matrices from components

    Args:
      - outer:   pointer to instance of calling object
      - x:       abscissa of data points ("x values")
      - y:       ordinate of data points ("y values")
      - ex:      independent uncertainties x
      - ey:      independent uncertainties y
      - erelx:   independent relative uncertainties x
      - erely:   independent relative uncertainties y
      - cabsx:   correlated abolute uncertainties x
      - crelx:   correlated relative uncertainties x
      - cabsy:   correlated absolute uncertainties y
      - crely:   correlated relative uncertainties y
      - quiet:   no informative printout if True

    Public methods:
      - get_Cov(): final covariance matrix (incl. proj. x)  
      - get_xCov(): covariance of x-values
      - get_yCov(): covariance of y-values
      - get_iCov(): inverse covariance matrix
      - plot(): provide a figure with data
 
    Data members:  
      * copy of all input arguments
      * covx: covariance matrix of x
      * covy: covariance matrix of y uncertainties
      * cov: full covariance matrix incl. projected x
      * iCov: inverse of covariance matrix
    """
    def __init__(self, outer,
                 x, y, ex, ey,
                 erelx, erely, cabsx, crelx, cabsy, crely,
                 quiet=True):

      self.needs_covariance = False # assume simple case w.o. cov.mat.

      nd = len(x)      
      # store input data as numpy float arrays, ensure length nd if needed
      self.x = np.asfarray(x)         # abscissa - "x values"
      self.y = np.asfarray(y)         # ordinate - "y values"
      if ex is not None:
        self.ex = np.asfarray(ex)       # independent uncertainties x
        if self.ex.ndim == 0:
          self.ex = self.ex * np.ones(nd)
        elif self.ex.ndim == 2:
          self.needs_covariance=True
      else:
        self.ex = None
      if ey is not None:
        self.ey = np.asfarray(ey)       # independent uncertainties y
        if self.ey.ndim == 0:
          self.ey = self.ey * np.ones(nd)
        elif self.ey.ndim == 2:
          self.needs_covariance=True
      else:
        self.ey = None
      if erelx is not None:
        self.erelx = np.asfarray(erelx) # independent relative uncertainties x
      else:
        self.erelx = None
      if erely is not None:
        self.erely = np.asfarray(erely) # independent relative uncertainties y
      else:
        self.erely = None
      if cabsx is not None:
        self.cabsx = np.asfarray(cabsx) # correlated abolute uncertainties x
        if self.cabsx.ndim == 0:
          self.cabsx = self.cabsx * np.ones(nd)
      else:
        self.cabsx = None
      if crelx is not None:   
        self.crelx = np.asfarray(crelx) # correlated relative uncertainties x
      else:
        self.crelx = None
      if cabsy is not None:
        self.cabsy = np.asfarray(cabsy) # correlated absolute uncertainties y
        if self.cabsy.ndim == 0:
          self.cabsy = self.cabsy * np.ones(nd)
      else:
        self.cabsy = None
      if crely is not None:
        self.crely = np.asfarray(crely) # correlated relative uncertainties y
      else:
        self.crely = None
      self.quiet = quiet      # no informative printout if True

      self.nd = nd
      self.model = None # no model defined yet

      # set flags for steering of fit process in do_fit()
      self.rebulildCov = None
      self.has_xErrors = ex is not None or erelx is not None \
        or cabsx is not None or crelx is not None
      self.has_rel_yErrors = erely is not None or crely is not None
      self.needs_covariance = self.needs_covariance or \
        self.cabsx is not None or self.crelx is not None or \
        self.cabsy is not None or self.crely is not None 


      # build (initial) covariance matrix (without x-errors)
      if self.needs_covariance:
        err2 = self._build_CovMat(self.nd,
                    self.ey, self.erely, self.cabsy, self.crely, self.y)
      else:
        err2 = self._build_Err2(self.ey, self.erely, self.y)

      # initialize uncertainties and covariance matrix,
      self._initialCov(err2)
      # sets: 
      #   self.covx: covariance matrix of x
      #   self.covy: covariance matrix of y uncertainties
      #   self.cov: full covariance matrix incl. projected x
      #   self.iCov: inverse of covariance matrix
      #   self.err2: array of squared uncertainties
      #   self.iErr2: 1./self.err2
      
    @staticmethod
    def _build_Err2(e=None, erel=None, data=None):
      """
      Build squared sum of independent absolute and/or 
      relative error components

      Args:
        * e: scalar or 1d np-array of float: independent uncertainties 
        * erel: scalar or 1d np-array of float: independent relative 
          uncertainties
      """
      
      err2 = 0.
      if e is not None:
        err2 = e * e
      if erel is not None:
        _er = erel * data
        err2 += _er * _er
      return err2

      
    @staticmethod
    def _build_CovMat(nd, e=None, erel=None,
                           eabscor=None, erelcor=None, data=None):
      """
      Build a covariance matrix from independent and correlated 
      absolute and/or relative error components

      Correlated absolute and/or relative uncertainties of input data 
      are to be specified as numpy-arrays of floats; they enter in the 
      diagonal and off-diagonal elements of the covariance matrix. 
      Values of 0. may be specified for data points not affected
      by a correlated uncertainty. E.g. the array [0., 0., 0.5., 0.5]
      results in a correlated uncertainty of 0.5 of the 3rd and 4th 
      data points.

      Covariance matrix elements of the individual components are added 
      to form the complete Covariance Matrix.
      
      Args:
        * nd: number of data points
        * e: 1d or 2d np-array of float: 
          independent uncertainties or a full covariance matrix
        * erel: 1d or 2d np-array of float:
          independent relative uncertainties or a full covariance matrix

      correlated components of uncertainties
        * eabscor: 1d np-array of floats or list of np-arrays:
        absolute correlated uncertainties
        * erelcor: 1d np-array of floats or list of np-arrays:
        relative correlated uncertainties
        * data: np-array of float: data, needed (only) for relative uncertainties

      Returns:
        * np-array of float: covariance matrix 
      """

      # 1. independent errors
      if e is not None:
        if e.ndim == 2: # already got a matrix, take as covariance matrix
          cov = np.array(e, copy=True)
        else:
          cov = np.diag(e * e) # set diagonal elements of covariance matrix
      else:
        cov = np.zeros( (nd, nd) )
    
      # 2. add relative errors
      if erel is not None:
        if erel.ndim == 2: # got a matrix
          cov += er * np.outer(data, data)
        else:
          er_ = np.array(erel) * data
          cov += np.diag(er_ * er_)   # diagonal elements of covariance matrix
        
      # 3. add absolute, correlated error components  
      if eabscor is not None:
        if len(np.shape(eabscor )) < 2: # has one entry
          cov += np.outer(eabscor, eabscor) 
        else:            # got a list, add each component
          for c in eabscor:
            cov += np.outer(c, c)
        
      # 4. add relative, correlated error components
      if erelcor is not None:
        if len(np.shape(erelcor) ) < 2: # has one entry
          c_ = erelcor * data
          cov += np.outer(c_, c_) 
        else:            # got a list, add each component
          for c in erelcor:
            c_ = np.array(c) * data
            cov += np.outer(c_, c_) 
      # return complete matrix
      return cov

    def _initialCov(self, err2):
      """Build initial (static) covariance matrix for y-errors
      (for pre-fit) and calculate inverse matrix
      """
      if err2.ndim == 2:
       # got a covariance matrix, need inverse
        self.needs_covariance = True
        self.covy = err2
        self.iCov = linalg.inv(err2)
        self.err2 = np.diagonal(err2) # squared diagonal elements
      else:
      # got independent uncertainties
        self.err2 = err2
        self.err2y = err2
        self.iErr2 = 1./err2
        self.covy = np.diag(err2)
        self.iCov = np.diag(1./self.err2)
      # do not rebuild covariance matrix in cost function
      self.needs_dynamicErrors = False 

      # no covariance of x-errors
      self.covx = None
      self.err2x = None
      # total covariance is that of y-errors
      self.cov = self.covy      
      
    def _init_dynamicErrors(self, ref_toModel = False, model = None):
      # method to switch on dynamic re-calculation of covariance matrix 
      self.ref_toModel = ref_toModel
      self.model = model

      self._staticCov = None
      self._staticErr2 = None
      self.iCov = None
      self.iErr2 = None

      # rebuild covariance matrix during fitting procedure
      self.needs_dynamicErrors = True    # flag for cost function
      self.final_call = False # flag for _rebuild_Cov: no storage of ycov 
      
      if self.needs_covariance:
        # build static (=parameter-independent) part of covariance matrix      
        if self.has_rel_yErrors and self.ref_toModel:
          # some y-errors are parameter-independent
          self._staticCov = self._build_CovMat(self.nd,
                       self.ey, eabscor = self.cabsy)
        else: 
          # all y-errors are parameter-independent
          self._staticCov = self._build_CovMat(self.nd,
                                  self.ey, erel=self.erely,
                                  eabscor = self.cabsy, erelcor=self.crely,
                                                data=self.y)
        # build matrix of relative errors
        if self.ref_toModel and self.has_rel_yErrors:
          self._covy0 = self._build_CovMat(self.nd,
                                  erel=self.erely,
                                  erelcor=self.crely,
                                  data=np.ones(self.nd))
        else:
          self._covy0 = None
        # covariance matrix of x-uncertainties (all are parameter-dependent)
        if self.has_xErrors:
          self.covx = self._build_CovMat(self.nd,
                              self.ex, self.erelx,
                              self.cabsx, self.crelx,
                              self.x)
         #  determine dx for derivative from smallest x-uncertainty
          self._dx = np.sqrt(min(np.diagonal(self.covx)))/10.
        else:
          self.covx = None

      else: # no covariance needed, use simple maths
        # build static (=parameter-independent) part of covariance matrix      
        if self.has_rel_yErrors and self.ref_toModel:
          # only independent y-errors do not depend on parameters
          self._staticErr2 = self._build_Err2(self.ey)
        else: 
          # all y-errors are parameter-independent
          self._staticErr2 = self._build_Err2( self.ey, self.erely, self.y)
          
        if self.has_xErrors:
          self.err2x = self._build_Err2(self.ex, self.erelx, self.x)
         #  determine dx for derivative from smallest x-uncertainty
          self._dx = np.sqrt(min(self.err2x))/10.
        else:
          self.err2x = None

    def _rebuild_Err2(self, mpar):
      """
      (Re-)calculate uncertaingies 
      """
      if self._staticErr2 is not None:
        self.err2 = np.array(self._staticErr2, copy=True)
      else:
        self.err2 = np.zeros(self.nd)
      if self.ref_toModel and self.has_rel_yErrors:
        _er = self.erely * self.model(self.x, *mpar)       
        self.err2 += _er * _er
      # remember y-errors  
      self.err2y = np.array(self.err2, copy=True)
     # add projected x errors
      if self.err2x is not None:
      # determine derivatives of model function w.r.t. x,
        _mprime = 0.5 / self._dx * (
               self.model(self.x + self._dx, *mpar) - 
               self.model(self.x - self._dx, *mpar) )
      # project on y and add to covariance matrix
        self.err2 += _mprime * _mprime * self.err2x
        
    def _rebuild_Cov(self, mpar):
      """
      (Re-)Build the covariance matrix from components
      and caclulate its inverse
      """
     # start from pre-built parameter-independent part of Covariance Matrix
      self.cov = np.array(self._staticCov, copy=True)

     # add matrix of parameter-dependent y-uncertainties
#      if self.ref_toModel and self.has_rel_yErrors:
      if self._covy0 is not None:
        _ydat = self.model(self.x, *mpar)       
        self.cov += self._covy0 * np.outer(_ydat, _ydat)
     # add projected x errors
      if self.has_xErrors:
        # store covariance matrix of y-uncertainties    
        if self.final_call:
          self.covy = np.array(self.cov, copy=True)

       # determine derivatives of model function w.r.t. x,
        _mprime = 0.5 / self._dx * (
               self.model(self.x + self._dx, *mpar) - 
               self.model(self.x - self._dx, *mpar) )
       # project on y and add to covariance matrix
        self.cov += np.outer(_mprime, _mprime) * self.covx      
      else: # no x-errors, y-covmat = covmat
        self.covy = self.cov    

    def get_Cov(self):
      """return covariance matrix of data
      """
      if self.needs_covariance:
        return self.cov
      else:
        if self.err2 is None:
          return None
        else:
          return np.diag(self.err2)
  
    def get_xCov(self):
      """return covariance matrix of x-data
      """
      if self.needs_covariance:
        return self.covx
      else:
        if self.err2x is None:
          return None
        else:
          return np.diag(self.err2x)

    def get_yCov(self):
      """return covariance matrix of y-data
      """
      if self.needs_covariance:
        return self.covy
      else:
        if self.err2y is None:
          return None
        else:
          return np.diag(self.err2y)
      
    def get_iCov(self):
      """return inverse of covariance matrix, as used in cost function
      """
      if self.needs_covariance:
        return self.iCov
      else:
        return np.diag(1./self.err2)

    def plot(self, num='xyData and Model',
                   figsize=(7.5, 6.5),                             
                   data_label='data' ):
      """return figure with xy data and uncertainties
      """
#    # get data
      x = self.x
      y = self.y
      ey = self.get_yCov()
      if ey.ndim == 2:
        ey = np.sqrt(np.diagonal(ey))
      else:
        ey = np.sqrt(ey)
      ex = self.get_xCov()
      if ex is not None:
        if ex.ndim ==2:
          ex = np.sqrt(np.diagonal(ex))
        else:
          ex = np.sqrt(ex)
   # draw data
      fig = plt.figure(num=num, figsize=figsize)
      plt.plot(x, y, marker='x', linestyle='', color='grey', alpha=0.5)
      if ex is not None:
        plt.errorbar(x, y, xerr=ex, yerr=ey, fmt='.', label=data_label)
      else:
        plt.errorbar(x, y, ey, fmt=".", label=data_label)
      return fig
      
  # define custom cost function for iminuit
  class xLSQ:
    """
    Custom e_x_tended Least-SQuares cost function with 
    dynamically updated covariance matrix and -2log(L) 
    correction term for parameter-dependent uncertainties

    For data points (x, y) with model f(x, \*p) 
    and covariance matrix V(f(x,\*p)
    the cost function is: 

    .. math:: 
      -2\ln {\cal L} = \chi^2(y, V^{-1}, f(x, *p) \,) 
      + \ln(\, \det( V(f(x, *p) ) \,)

    For uncertainties depending on the model parameters, a more
    efficient approach is used to calculate the likelihood, which
    uses the Cholesky decompostion of the covariance matrix into a
    product of a triangular matrix and its transposed

    .. math::
       V = L L^T,

    thus avoiding the costy calculation of the inverse matrix.
    
    .. math::
      \chi^2 = {r}\cdot (V^{-1}{r}) ~~with~~ r = y - f(x,*p)

    is obtained by solving the linear equation  

   .. math::
      V X = r, ~i.e.~ X=V^{-1} r ~and~ \chi^2= r \cdot X   

   with the effecient linear-equation solver *scipy.linalg.cho_solve(L,x)*
   for Cholesky-decomposed matrices.

   The determinant is efficiently calculated by taking the product 
   of the diagonal elements of the matrix L,

    .. math::
      \det(V) = 2 \, \prod L_{i,i}
    
    Input:

    - outer: pointer to instance of calling class
    - data: data object of type xyDataUncertainties
    - model: model function f(x, \*par)
    - use_neg2logL: use full -2log(L) instead of chi2 if True

    __call__ method of this class is called by iminuit

    Data members:

    - ndof: degrees of freedom 
    - nconstraints: number of parameter constraints
    - gof: chi2-value (goodness of fit)
    - use_neg2logL: usage of full 2*neg Log Likelihood
    - quiet: no printpout if True    

    Methods:

    - model(x, \*par)
    """
 
    def __init__(self, outer, 
                 data, model,
                 use_neg2logL=False, quiet=True):

      from iminuit.util import make_func_code

      self.data = data
      self.model = model
      self.quiet = quiet
      # use -2 * log(L) of Gaussian instead of Chi2
      #  (only different from Chi2 for parameter-dependent uncertainties)
      self.use_neg2logL = use_neg2logL
      
      # set proper signature of model function for iminuit
      self.pnams = outer.pnams
      self.func_code = make_func_code(self.pnams)
      self.npar = len(self.pnams)
      # dictionary assigning parameter name to index
      self.pnam2id = {
        self.pnams[i] : i for i in range(0,self.npar)
        } 
      self.ndof = len(data.y) - self.npar
      self.constraints = []
      self.nconstraints = 0
      # flag to control final actions in cost function
      self.final_call = False

    def setConstraints(self, constraints):
      """Add parameter constraints

      format: nested list(s) of type 
      [parameter name, value, uncertainty] or
      [parameter index, value, uncertainty]
      """
      
      if isinstance(constraints[1], list):
         for c in constraints:
           self.constraints.append(c)
      else:
         self.constraints.append(constraints)
      self.nconstraints = len(self.constraints)
      # take account of constraints in degrees of freedom 
      self.ndof = len(self.data.y) - self.npar + self.nconstraints

    def __call__(self, *par):  
      # called iteratively by minuit

      # cost funtion is extended chi2:
      #   add normalsation term if uncertainties depend on model 

      nlL2 = 0. # initialize -2*ln(L)
      #  first, take into account possible parameter constraints  
      if self.nconstraints:
        for c in self.constraints:
          if type(c[0])==type(' '):
            p_id = self.pnam2id[c[0]]
          else:
            p_id = c[0]
          r = ( par[p_id] - c[1]) / c[2] 
          nlL2 += r*r

      # calculate residual of data wrt. model    
      _r = self.data.y - self.model(self.data.x, *par)

      if self.data.needs_covariance:
        #  check if matrix needs rebuilding
        if not self.data.needs_dynamicErrors:
         # static covariance, use its inverse
          nlL2 += float(np.inner(np.matmul(_r, self.data.iCov), _r))
          # identical to classical Chi2
          self.gof = nlL2
          
        else: # dynamically rebuild covariance matrix
          self.data._rebuild_Cov(par)
          # use Cholesky decompositon to compute chi2 = _r.T (V^-1) _r 
          L, is_lower = linalg.cho_factor(self.data.cov, check_finite=False)
          nlL2 += np.inner(_r, linalg.cho_solve((L, is_lower), _r) )
          # up to here, identical to classical Chi2
          self.gof = nlL2                  
        # take into account parameter-dependent normalisation term
          if self.use_neg2logL:
         #  fast calculation of determinant det(V) from Cholesky factor 
            nlL2 += 2.*np.sum(np.log(np.diagonal(L) ) )

      else:  # fast calculation for simple errors
        # check if errors needs recalculating
        if self.data.needs_dynamicErrors:
          self.data._rebuild_Err2(par)
          nlL2 += np.sum(_r * _r / self.data.err2)
        else:
          nlL2 += np.sum(_r * _r * self.data.iErr2)

        # this is identical to classical Chi2
        self.gof = nlL2
        
        # add parameter-dependent normalisation term if needed and wanted
        if self.data.needs_dynamicErrors and self.use_neg2logL:
          nlL2 += np.sum(np.log(self.data.err2))

      return nlL2
    
  # --- end definition of class xLSQ ----

  #
  # --- code for histogram Fit
  #      

  def set_hOptions(self,
              run_minos=None,
              use_GaussApprox=None,
              fit_density = None,
              quiet=None):

    """Define mnFit options

       Args:
        - run minos else don*t run minos
        - use Gaussian Approximation of Poisson distribution
        - don*t provide printout else verbose printout 
    """
    if run_minos is not None:   
      self.run_minos = run_minos
    if use_GaussApprox is not None:   
      self.use_GaussApprox = use_GaussApprox
    if fit_density is not None:
      self.fit_density = fit_density
    if quiet is not None:
      self.quiet = quiet
    
  def init_hData(self,
                bin_contents, bin_edges, DeltaMu=None):
    """
    initialize histogram data object

    Args:
    - bin_contents: array of floats
    - bin_edges: array of length len(bin_contents)*1
    - DeltaMu: shift in mean (Delta mu) versus lambda 
    of Poisson distribution 
    """
    
    # create data object and pass all input arguments
    self.hData = self.histData(self,
                               bin_contents, bin_edges,
                               DeltaMu,
                               quiet=self.quiet)
    self.data = self.hData
    
  def init_hFit(self, model, p0=None,
                constraints=None,
                limits=None):
    """initialize fit object

    Args:
      - model: model density function f(x; \*par)
      - p0: np-array of floats, initial parameter values 
      - constraints: (nested) list(s): [parameter name, value, uncertainty] 
        or [parameter index, value, uncertainty]
      - limits: (nested) list(s): [parameter name, min, max] 
        or [parameter index, min, max]
    """
    
    # get parameters of model function to set start values for fit
    args, model_kwargs = self.get_functionSignature(model)
    self.pnams = list.copy(list(model_kwargs.keys()))
    if p0 is not None:
      for i, pnam in enumerate(self.pnams):
        model_kwargs[pnam] = p0[i]
        
    # create cost function
    self.costf = self.hCost(self,
                            self.hData, model,
                            use_GaussApprox=self.use_GaussApprox,
                            density = self.fit_density,
                            quiet=self.quiet)
    if limits is not None:
      self.setLimits(limits)

    if constraints is not None:
      self.costf.setConstraints(constraints)

    # create Minuit object
    if __version__ < '2':
      if self.quiet:
        print_level = 0
      else:
        print_level = 1
      if limits is not None:
        for i, pnam in enumerate(self.pnams):
          model_kwargs['limit_' + pnam] = self.limits[i]
          
      self.minuit = Minuit(self.costf, 
                           errordef=self.ErrDef,
                           print_level=print_level,
                           **model_kwargs )
    else:
      self.minuit = Minuit(self.costf, **model_kwargs)  
      self.minuit.errordef = self.ErrDef
      if self.quiet:
        self.minuit.print_level = 0
      if limits is not None:
        self.minuit.limits = self.limits       

  def do_hFit(self):
    """perform fit sequence for histogram fit
    """
    if self.hData is None:
      print(' !!! mnFit: no data object defined - call init_data()')
      sys.exit('mnFit Error: no data object')
    if self.costf is None:
      print(' !!! mnFit: no fit object defined - call init_fit()')
      sys.exit('mnFit Error: no fit object')
    
    # summarize options
    if not self.quiet:
      print( '*==* mnFit starting (pre-)fit')
      print( '  Options:')
      if self.use_GaussApprox:
        print( '     - using Gaussian approximation of Poisson distibution')
      if self.run_minos is not None:
        print( '     - performing minos profile likelihood scan')
        
    # perform fit
    try:
      self.migradResult = self.minuit.migrad()  # find minimum of cost function
      self.migrad_ok = True
    except Exception as e:
      self.migrad_ok = False
      print('*==* !!! fit with migrad failed')
      print(e)
      exit(1)

    # run profile likelihood scan to check for asymmetric errors
    if self.run_minos:
      if not self.quiet:
        print( '*==* mnFit starting minos scan')
      try:  
        self.minosResult = self.minuit.minos()
        self.minos_ok = True
      except Exception as e:
        self.minos_ok = False
        if not self.quiet:
          print( '*==* mnFit: !!! minos failed \n', e)
        
    self._storeResult()
    
    return self.migradResult, self.minosResult

  
  class histData:
    """
      Container for Histogram data

      Data Members:

      - contents, array of floats: bin contents
      - edges, array of floats: bin edges (nbins+1 values)

      calculated from input:

      - nbins: number of bins
      - lefts: left edges
      - rights: right edges
      - centers: bin centers
      - widths: bin widths
      - Ntot: total number of entries, used to normalize probatility density

      available after completion of fit:

      - model_values: bin contents from best-fit model 
      - model_related_uncertainties: uncertainties fom best-fit model_values

      Methods:

      - plot(): create figure with histogram of data and uncertainties
    """
    
    def __init__(self, outer,
                 bin_contents, bin_edges, DeltaMu=None, quiet=True):
      """ 
      initialize histogram Data

      Args:
      - bin_contents: array of floats
      - bin_edges: array of length len(bin_contents)*1
      - DeltaMu: array of floats, shift of mean mu vs. 
          lambda of Poisson distribution, DeltaMu = mu-lambda 
      - quiet: boolean, controls printed output

      """

      self.contents = bin_contents
      self.nbins=len(bin_contents)
      self.Ntot = np.sum(bin_contents)
      self.edges = bin_edges
      if DeltaMu is None:
        self.DeltaMu = np.zeros(len(bin_contents))
      else:
        self.Delta = DeltaMu
      #  
      self.lefts=self.edges[:-1]
      self.rights=self.edges[1:]
      self.centers = (self.rights  + self.lefts)/2.
      self.widths = self.rights - self.lefts
      # flag to control final actions in cost function
      self.final_call = False
      self.model_values = None
      self.model_related_uncertainties = None
      
    def plot(self, num='histData and Model',
                   figsize=(7.5, 6.5),                             
                   data_label='Binned data' ):
      """return figure with histogram data and uncertainties
      """

      w = self.edges[1:] - self.edges[:-1]
      fig = plt.figure(num=num, figsize=figsize)
      if self.model_values is not None:
        plt.bar(self.centers, self.model_values,
              align='center', width = w,
              facecolor='wheat', edgecolor='brown', alpha=0.2,
                label = "entries/bin from model")
      else:
        plt.bar(self.centers, self.contents,
              align='center', width = w,
              facecolor='cadetblue', edgecolor='darkblue', alpha=0.2,
              label = data_label)
      # set and plot error bars
      if self.model_related_uncertainties is not None:
        ep = self.model_related_uncertainties
      else:
        ep = (self.contents + np.abs(self.DeltaMu))      
      em = [ep[i] if self.contents[i]-ep[i]>0. else self.contents[i] for i in range(len(ep))]
      plt.errorbar(self.centers, self.contents,
                   yerr=(em, ep),
                   fmt='_', color='darkblue', markersize=15,
                   ecolor='darkblue', alpha=0.8,
                   label = data_label)
      return fig
      
  # --- cost function for histogram data
  class hCost:
    """
    Cost function for binned data

    The __call__ method of this class is called by iminuit.

    The default cost function to minimoze is twice the negative 
    log-likelihood of the Poisson distribution generalized to 
    continuous observations x by replacing k! by the gamma function:

    .. math::
        cost(x;\lambda) = 2 \lambda (\lambda - x*\ln(\lambda) + \ln\Gamma(x+1.))

    Alternatively, the Gaussian approximation is available:

    .. math::
        cost(x;\lambda) = (x - \lambda)^2 / \lambda + \ln(\lambda)
           
    The implementaion also permits to shift the obervation x by an
    offset to take into account corrections to the number of observed
    bin entries (e.g. due to background or efficiency corrections):
    x -> x-deltaMu with deltaMu = mu - lambda, where mu is the mean
    of the shifted Poisson or Gauß distibution.  

    Input:

    - outer: pointer to instance of calling class
    - hData: data object of type histData
    - model: model function f(x, \*par)
    - use_GaussApprox, bool: use Gaussian approximation 
    - density, bool: fit a normalised density; if false, an overall
      normalisation must be provided in the model function

    Data members:

    - ndof: degrees of freedom 
    - nconstraints: number of parameter constraints
    - gof: goodness-of-fit as likelihood ratio w.r.t. the 'saturated model'

    External references:

    - model(x, \*par): the model function 
    - data: pointer to instance of class histData
    - data.model_values: bin entries calculated by the best-fit model
    - data.model_related_uncertainties: uncertainties calulated from 
      best-fit model_values
    """

    def __init__(self, outer, 
                 hData, model,
                 use_GaussApprox=False, density= True,
                 quiet=True):
      from iminuit.util import make_func_code

      self.data = hData
      self.model = model
      self.density = density
      self.GaussApprox = use_GaussApprox
      self.quiet = quiet
      
      # set proper signature of model function for iminuit
      self.pnams = outer.pnams
      self.func_code = make_func_code(self.pnams)
      self.npar = len(self.pnams)
      # dictionary assigning parameter name to index
      self.pnam2id = {
        self.pnams[i] : i for i in range(0,self.npar) } 
      self.ndof = len(self.data.contents) - self.npar
      self.constraints = []
      self.nconstraints = 0
      # flag to control final actions in cost function
      self.final_call = False

      if self.GaussApprox:
        self.n2lLcost = self.n2lLGauss
      else:
        self.n2lLcost = self.n2lLPoisson

      if self.density:
        self.norm = self.data.Ntot
      else:
        self.norm = 1.
        
    def setConstraints(self, constraints):
      """
      Add parameter constraints

      format: nested list(s) of type 
      [parameter name, value, uncertainty] or
      [parameter index, value, uncertainty]
      """
      
      if isinstance(constraints[1], list):
         for c in constraints:
           self.constraints.append(c)
      else:
         self.constraints.append(constraints)
      self.nconstraints = len(self.constraints)
      # take account of constraints in degrees of freedom 
      self.ndof = len(self.data.contents) - self.npar + self.nconstraints
    
    def __call__(self, *par):
      # called iteratively by minuit

      # cost function is likelihood of shifted poisson or Gauss approximation

      # - first, take into account possible parameter constraints  
      n2lL= 0.
      if self.nconstraints:
        for c in self.constraints:
          if type(c[0])==type(' '):
            p_id = self.pnam2id[c[0]]
          else:
            p_id = c[0]
          r = ( par[p_id] - c[1]) / c[2] 
          n2lL += r*r

      # - calculate 2*negLogL Poisson;
      #  model prediction as appr. integral over bin
      model_values = self.norm * self.integral_overBins(
        self.data.lefts, self.data.rights,
        self.model, *par) 
      # 

      n2lL += np.sum(
        self.n2lLcost( self.data.contents - self.data.DeltaMu, 
                      model_values + np.abs(self.data.DeltaMu) ) )
       
      if self.final_call:
        # store goodness-of-fit (difference of nlL2 w.r.t. saturated model)
        n2lL_saturated = np.sum(
          self.n2lLcost(
              self.data.contents - self.data.DeltaMu, 
              self.data.contents + np.abs(self.data.DeltaMu) + 0.005) )
        #                                !!! const. 0.005 to aviod log(0.)
        self.gof =  n2lL - n2lL_saturated

        # provide model values and model-related uncertainties to data object
        self.data.model_values = model_values       
        self.data.model_related_uncertainties = np.sqrt( 
                           model_values + np.abs(self.data.DeltaMu) ) 
        
       # return 2 * neg. logL
      return n2lL
    
    @staticmethod
    def n2lLPoisson(x, lam):  
      """
      neg. logarithm of Poisson distribution for real-valued x

      """
      return 2.*(lam - x*np.log(lam) + loggamma(x+1.))

    @staticmethod
    def n2lLsPoisson(xs, lam, mu):  
      """
      2* neg. logarithm of generalized Poisson distribution: 
      shifted to new mean mu for real-valued xk        
      for lam=mu, the standard Poisson distribution is recovered
      lam=sigma*2 is the variance of the shifted Poisson distribution.
      """
      xs = (xk + lam - mu)
      return 2.*(lam - xs*np.log(lam) + loggamma(xs+1.))

    @staticmethod
    def n2lLGauss(x, lam):  
      """    
      negative log-likelihood of Gaussian approximation
      Pois(x, lam) \simeq Gauss(x, mu=lam, sigma^2=lam)
      """
      r = (x-lam)
      return (r * r/lam + np.log(lam))
           
    @staticmethod
    def integral_overBins(ledges, redges, f, *par):
      """Calculate approx. integral of model over bins using Simpson's rule
      """
      return (redges - ledges)/6. * \
                            ( f(ledges, *par) \
                            + 4.*f((ledges+redges)/2., *par) \
                            + f(redges, *par) ) 

  # --- end definition of class hCost ----

  #
  # --- code for fit with user-supplied cost function
  #      
  def init_mnFit(self, userCostFunction, p0=None, 
                       constraints=None, limits=None):
    """initialize fit object for simple minuit fit with user-supplied cost

    Args:
      - costFunction: cost function to optimize
      - p0: np-array of floats, initial parameter values 
      - limits: (nested) list(s): [parameter name, min, max] 
        or [parameter index, min, max]
    """

    # get parameters of model function to set start values for fit
    args, model_kwargs = self.get_functionSignature(userCostFunction)
    self.pnams = list.copy(list(model_kwargs.keys()))
    if p0 is not None:
      for i, pnam in enumerate(self.pnams):
        model_kwargs[pnam] = p0[i]
        
    #set up cost function for iminuit ...
    self.costf = self.mnCost(self,
                             userCostFunction,
                             quiet=self.quiet)      
    if limits is not None:
      self.setLimits(limits)

    if constraints is not None:
      self.costf.setConstraints(constraints)

    # ... and create Minuit object
    if __version__ < '2':
      if self.quiet:
        print_level = 0
      else:
        print_level = 1
      if limits is not None:
        for i, pnam in enumerate(self.pnams):
          model_kwargs['limit_' + pnam] = self.limits[i]
          
      self.minuit = Minuit(self.costf, 
                           errordef=self.ErrDef,
                           print_level=print_level,
                           **model_kwargs )
    else:
      self.minuit = Minuit(self.costf, **model_kwargs)  
      self.minuit.errordef = self.ErrDef
      if self.quiet:
        self.minuit.print_level = 0
      if limits is not None:
        self.minuit.limits = self.limits
        
  def set_mnOptions(self,
                      run_minos=None,
                      neg2logL=None,
                      quiet=None):
    """Define options for minuit fit with user cost function

    Args:

    - run_minos: run minos profile likelihood scan
    - neg2logL: cost function is -2 negLogL

    """
    if run_minos is not None:   
      self.run_minos = run_minos
    if neg2logL is not None:
      self.neg2logL = neg2logL
      if self.neg2logL:
        self.ErrDef = 1.
      else:
        self.ErrDef = 0.5
    if quiet is not None:
      self.quiet = quiet

  def do_mnFit(self):
    """perform fit sequence for user-defined cost function
    """
    if self.costf is None:
      print(' !!! mnFit: no fit object defined - call init_fit()')
      sys.exit('mnFit Error: no fit object')
    
    # summarize options
    if not self.quiet:
      print( '*==* mnFit starting (pre-)fit')
      print( '  Options:')
      if self.neg2logL:
        print( ' assuming cost is 2 * negative log-likelihood')
      else:
        print( ' assuming cost is negative log-likelihood')
        
    # perform fit
    try:
      self.migradResult = self.minuit.migrad()  # find minimum of cost function
      self.migrad_ok = True
    except Exception as e:
      self.migrad_ok = False
      print('*==* !!! fit with migrad failed')
      print(e)
      exit(1)

    # run profile likelihood scan to check for asymmetric errors
    if self.run_minos:
      if not self.quiet:
        print( '*==* mnFit starting minos scan')
      try:  
        self.minosResult = self.minuit.minos()
        self.minos_ok = True
      except Exception as e:
        self.minos_ok = False
        if not self.quiet:
          print( '*==* mnFit: !!! minos failed \n', e)        
    self._storeResult()
    return self.migradResult, self.minosResult

  # --- class encapsulating user-defined cost function
  class mnCost:
    """
    Interface for simple minuit fit with user-supplied cost function.

    The __call__ method of this class is called by iminuit.

    Args:

      - userCostFunction: user-supplied cost function for minuit;
         must be a negative log-likelihood 
    """
    
    def __init__(self, outer, 
                 userCostFunction,
                 quiet=True):
      from iminuit.util import make_func_code

      self.cost = userCostFunction
      self.quiet = quiet

      self.ErrDef = outer.ErrDef

      # set proper signature of model function for iminuit
      self.pnams = outer.pnams
      self.func_code = make_func_code(self.pnams)
      self.npar = len(self.pnams)
      # dictionary assigning parameter name to index
      self.pnam2id = {
        self.pnams[i] : i for i in range(0,self.npar) } 
      self.constraints = []
      self.nconstraints = 0

      # for this kind of fit, some input and ouput quantities are not know
      self.data = None
      self.gof = None
      self.ndof = None
      
    def setConstraints(self, constraints):
      """
      Add parameter constraints

      format: nested list(s) of type 
      [parameter name, value, uncertainty] or
      [parameter index, value, uncertainty]
      """
      
      if isinstance(constraints[1], list):
         for c in constraints:
           self.constraints.append(c)
      else:
         self.constraints.append(constraints)
      self.nconstraints = len(self.constraints)

    def __call__(self, *par):
      cost = 0.
      # add constraints to cost
      if self.nconstraints:
        for c in self.constraints:
          if type(c[0])==type(' '):
            p_id = self.pnam2id[c[0]]
          else:
            p_id = c[0]
          r = ( par[p_id] - c[1]) / c[2] 
          cost += r*r
        cost *= self.ErrDef 
      # called iteratively by minuit
      return cost + self.cost(*par)

 # --- end definition of class mnCost ----

 #
 # --- comon code for all fit types
 #
  def _storeResult(self):
  # collect results as numpy arrays
    # !!! this part depends on iminuit version !!!    
    m=self.minuit
    minCost = m.fval                        # minimum value of cost function
    npar = m.nfit                           # numer of parameters
    ndof = self.costf.ndof                  # degrees of freedom
    if __version__< '2':
      parnames = m.values.keys()            # parameter names
      parvals = np.array(m.values.values()) # best-fit values
      parerrs = np.array(m.errors.values()) # parameter uncertainties
      cov=np.array(m.matrix())
    else:
    # vers. >=2.0 
      parnames = m.parameters      # parameter names
      parvals = np.array(m.values) # best-fit values
      parerrs = np.array(m.errors) # parameter uncertainties
      cov=np.array(m.covariance)
      
    if self.minosResult is not None and self.minos_ok:
      pmerrs = [] 
    #  print("MINOS errors:")
      if __version__< '2':
        for pnam in m.merrors.keys():
          pmerrs.append([m.merrors[pnam][2], m.merrors[pnam][3]])
      else:
        for pnam in m.merrors.keys():
          pmerrs.append([m.merrors[pnam].lower, m.merrors[pnam].upper])
      self.OneSigInterval=np.array(pmerrs)
    else:
      self.OneSigInterval = np.array(list(zip(-parerrs, parerrs)))

    # final call of cost function at miminum to update all results
    # -  signals data object to store model-dendent uncertainties
    if self.data is not None: self.data.final_call=True
    # -  and cost function to store goodness-of-fit
    self.costf.final_call=True
    fval = self.costf(*parvals) 
  
    # store results as class members
    #   parameter names
    self.ParameterNames = parnames
    #   chi2 at best-fit point (possibly different from minCost)
    self.GoF = self.costf.gof  
    #   parameter values at best-fit point
    self.ParameterValues = np.array(parvals, copy=True)
    #   number of degrees of freedom
    self.NDoF = ndof  
    #   symmetric uncertainties
    self.MigradErrors = np.array(parerrs, copy=True)
    #   covariance and correlation matrices
    self.CovarianceMatrix = np.array(cov, copy=True)
    self.CorrelationMatrix = cov/np.outer(parerrs, parerrs)
    #   1-sigma (68% CL) range in self.OneSigInterval
    
  def getResult(self):
    """return most im portant results as numpy arrays
    """
    return (self.ParameterValues, self.OneSigInterval,
            self.CorrelationMatrix, self.GoF)

  @staticmethod
  def getFunctionError(x, model, pvals, covp):
    """ determine error of model at x  
    
    Formula: 
      Delta(x) = sqrt( sum_i,j (df/dp_i(x) df/dp_j(x) Vp_i,j) )

    Args:
      * x: scalar or np-array of x values
      * model: model function
      * pvlas: parameter values
      * covp: covariance matrix of parameters

    Returns:
      * model uncertainty, same length as x
    """

    # calculate partial derivatives of model w.r.t parameters    

    #   parameter step size 
    dp = 0.01 * np.sqrt(np.diagonal(covp))
    #   derivative df/dp_j at each x_i
    dfdp = np.empty( (len(pvals), len(x)) )
    p_plus = np.array(pvals, copy=True)
    p_minus = np.array(pvals, copy=True)
    for j in range(len(pvals)): 
      p_plus[j] = pvals[j] + dp[j]
      p_minus[j] = pvals[j] - dp[j]
      dfdp[j] = 0.5 / dp[j] * (
                    model(x, *p_plus) - 
                    model(x, *p_minus) )
      p_plus[j] = pvals[j]
      p_minus[j] = pvals[j]
    #   square of uncertainties on function values
    Delta= np.empty(len(x))
    for i in range(len(x)):
      Delta[i] = np.sum(np.outer(dfdp[:,i], dfdp[:,i]) * covp)
    return np.sqrt(Delta) 
  
  def plotModel(self,
                axis_labels=['x', 'y = f(x, *par)'], 
                data_legend = 'data',    
                model_legend = 'fit',
                plot_band=True): 
    """
    Plot model function and data 
    
    Uses iminuitObject, cost Fuction (and data object)

    Args: 
      * list of str: axis labels
      * str: legend for data
      * str: legend for model 

    Returns:
      * matplotlib figure
    """
    
  # access low-level fit objects
    m = self.minuit  # minuit object
    cf = self.costf  # cost function object
    d = cf.data
    
  # retrieve fit results
    pvals, pmerrs, cor, gof = self.getResult()
    # symmetric errors
    perrs = (pmerrs[:,1]-pmerrs[:,0])/2.
    # covariance matrix
    pcov = cor * np.outer(perrs, perrs)
    pnams = self.ParameterNames
    ndof = cf.ndof
    chi2prb = self.chi2prb(gof, ndof) 
    
  # plot data
    fig_model = d.plot(figsize=(7.5, 6.5),
          data_label=data_legend)

  # overlay model function
    # histogram fit provides normalised distribution,
    #    determine bin widhts and scale factor
    xmin, xmax = plt.xlim()    
    xplt = np.linspace(xmin, xmax, 190)
    if self.fit_type=='hist':
      # detemine local bin width
      bwidths = np.zeros(len(xplt))
      i=0
      for j, x in enumerate(xplt):
        if x >= cf.data.rights[min(i, cf.data.nbins-1)]:
          i += 1
        bwidths[j] = cf.data.widths[min(i, cf.data.nbins-1)]
      sfac = cf.norm * bwidths
    elif self.fit_type=="xy":
      sfac = 1.
    else:
      print("!**! mnFit.plotModel: unknown fit type: self.fit_type")
      sfac = 1.
    # plot model line  
    yplt = cf.model(xplt, *pvals)
    plt.plot(xplt, yplt*sfac, label=model_legend,
             linestyle='dashed', alpha=0.7,
             linewidth=2.5, color='darkorange')
    plt.xlabel(axis_labels[0], size='x-large')
    plt.ylabel(axis_labels[1], size='x-large')
    plt.grid()
   # draw error band around model function
    if plot_band:
      DeltaF = self.getFunctionError(xplt, cf.model, pvals, pcov)
      plt.fill_between(xplt, sfac*(yplt+DeltaF),
                             sfac*(yplt-DeltaF),
                             alpha=0.3, color='darkkhaki')
      plt.plot(xplt, sfac*(yplt+DeltaF), linewidth=1, 
                             alpha=0.4, color='darkgreen')
      plt.plot(xplt, sfac*(yplt-DeltaF), linewidth=1,
                             alpha=0.4, color='darkgreen')

  # display legend with some fit info
    fit_info = []
    #  1. parameter values and uncertainties
    pe = 2   # number of significant digits of uncertainty
    if self.minosResult is not None and self.minos_ok:
      for pn, v, e in zip(pnams, pvals, pmerrs):
        nd, _v, _e = self.round_to_error(v, min(abs(e[0]), abs(e[1])),
                                         nsd_e=pe )
        txt="{} = ${:#.{pv}g}^{{+{:#.{pe}g}}}_{{{:#.{pe}g}}}$"
        fit_info.append(txt.format(pn, _v, e[1], e[0], pv=nd, pe=pe))
    else:
      for pn, v, e in zip(pnams, pvals, pmerrs):
        nd, _v, _e = self.round_to_error(v, e[1], nsd_e=pe)
        txt="{} = ${:#.{pv}g}\pm{:#.{pe}g}$"
        fit_info.append(txt.format(pn, _v, _e, pv=nd, pe=pe))
    #  2. goodness-of-fit
    if self.fit_type=='xy':
      fit_info.append(
        "$\\chi^2$/$n_\\mathrm{{dof}}$={:.1f}/{}".format(gof,ndof) + \
         ", p={:.1f}%".format(100*chi2prb) )
    elif self.fit_type=='hist':
      fit_info.append(
        "g.o.f./$n_\\mathrm{{dof}}$ = {:.1f}/{}".format(gof, ndof) )
    #  add legend to plot  
    plt.legend(loc='best', title="\n".join(fit_info))      

    return fig_model
  
# plot array of profiles and contours
  def plotContours(self, figname='Profiles and Contours'):
    """
    Plot grid of profile curves and one- and two-sigma
    contour lines from iminuit object

    Arg: 
      * iminuitObject

    Returns:
      * matplotlib figure 
    """

    if not self.quiet:
      print( '*==* mnFit: scanning contours')

    m = self.minuit     
    npar = m.nfit    # numer of parameters
    if __version__< '2':
      pnams = m.values.keys()  # parameter names
    else:
  # vers. >=2.0 
      pnams = m.parameters      # parameter names

    fsize=3.5
    cor_fig, axarr = plt.subplots(npar, npar,
                                  num=figname,
                                  figsize=(fsize*npar, fsize*npar))
# protect the following, may fail
    try:
      ip = -1
      for i in range(0, npar):
        ip += 1
        jp = -1
        for j in range(0, npar):
          jp += 1
          if ip > jp:
           # empty space
            axarr[jp, ip].axis('off')
          elif ip == jp:
           # plot profile
            plt.sca(axarr[ip, ip])
            m.draw_mnprofile(pnams[i], subtract_min=True)
            plt.ylabel('$\Delta\chi^2$')
          else:
            plt.sca(axarr[jp, ip])
            if __version__ <'2':
              m.draw_mncontour(pnams[i], pnams[j])
            else:
              m.draw_mncontour(pnams[i], pnams[j],
                cl=(self.Chi22CL(1.), self.Chi22CL(4.)) )
      return cor_fig

    except Exception as e:
      print('*==* !!! profile and contour scan failed')
      print(e)
      return None


  def plot_Profile(self, pnam):
    """plot profile likelihood of parameter pnam
    """
    fig = plt.figure(num='Likelihood profile ' + pnam,
                     figsize=(5., 5.))
    self.minuit.draw_mnprofile(pnam, subtract_min=True)
    return fig


  def plot_clContour(self, pnam1, pnam2, cl):
    """plot a contour of parameters pnam1 and pnam2
    with confidence level(s) cl
    """
    if __version__ <'2':
      print("!!! plot_clContour not implemented vor iminuit vers.<2")
      return
    else:
      fig = plt.figure(num='Contour(s) ' + pnam1 + ' vs. ' + pnam2,
                       figsize=(5., 5.))
      self.minuit.draw_mncontour(pnam1, pnam2, cl=cl)    
      return fig


  def plot_nsigContour(self, pnam1, pnam2, nsig):
    """plot nsig contours of parameters pnam1 and pnam2
    """
    fig = plt.figure(num='Contour(s) ' + pnam1 + ' vs. ' + pnam2,
      figsize=(5., 5.))
    if __version__ <'2':
      self.minuit.draw_mncontour(pnam1, pnam2, nsigma=nsig)
    else:
      ns = range(1, nsig+1)
      dchi2 = np.array(ns)**2
      cl = self.Chi22CL(dchi2)    
      self.minuit.draw_mncontour(pnam1, pnam2, cl=cl)    
    return fig

  @staticmethod
  def round_to_error(val, err, nsd_e=2):
    """round float *val* to same number of sigfinicant digits as uncertainty *err*
  
    Returns:
      * int:   number of significant digits for v
      * float: val rounded to precision of err
      * float: err rounded to precision nsd_e

    """

    v = abs(val)
    # round uncertainty to nd0 significant digits
    e = float("{:.{p}g}".format(abs(err), p=nsd_e))
    
    # determine # of siginifcant digits vor v
    _nd = int( np.floor(np.log10(v) - np.floor(np.log10(e)) ) ) + nsd_e
    # take into account possible rounding of v ...
    v = float("{:.{p}g}".format(v, p=_nd))
    # ... and determine final # of sig. digits
    nsd_v = int( np.floor(np.log10(v) - np.floor(np.log10(e)) ) ) + nsd_e
    v = float("{:.{p}g}".format(v, p=nsd_v))
     
    return nsd_v, np.sign(val)*v, e               

  @staticmethod
  def chi2prb(chi2,ndof):
    """Calculate chi2-probability from chi2 and degrees of freedom
    """
    return 1.- stats.chi2.cdf(chi2, ndof)
  
  @staticmethod
  def CL2Chi2(CL):
    """calculate DeltaChi2 from confidence level CL for 2-dim contours
    """
    return -2.*np.log(1.-CL)

  @staticmethod
  def Chi22CL(dc2):
    """calculate confidence level CL from DeltaChi2 for 2-dim contours
    """
    return (1. - np.exp(-dc2 / 2.))

  @staticmethod
  def get_functionSignature(f):
    """get arguments and keyword arguments passed to a function
    """
    pars = signature(f).parameters
    args = []
    kwargs = {}
    for p in pars.values():
      if p.default is p.empty:
        args.append(p.name)
      else:
        kwargs[p.name]=p.default
    return args, kwargs
    

if __name__ == "__main__": # --- interface and example
  

  def example_xyFit():
  #
  # *** Example of an application of phyFit.mFit()
  #
  # define the model function to fit
    def exp_model(x, A=1., x0=1.):
      return A*np.exp(-x/x0)

    # another model function
    def poly2_model(x, a=0.1, b=1., c=1.):
      return a*x**2 + b*x + c

    # set model to use in fit
    fitmodel=exp_model  # also try poly2_model !
    # get keyword-arguments
    mpardict = mnFit.get_functionSignature(fitmodel)[1]
  
    # the data ...
    data_x = [0.0, 0.2, 0.4, 0.6, 0.8, 1., 1.2,
          1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6]
    data_y = [1.149, 0.712, 0.803, 0.464, 0.398, 0.354, 0.148,
            0.328, 0.181, 0.140, 0.065, 0.005,-0.005, 0.116]
    # ... and uncertaities  
    sabsy = 0.07 # independent y
    srely = 0.05 # 5% of model value
    cabsy = 0.04 # correlated
    crely = 0.03 # 3% of model value correlated
    sabsx = 0.05 # independent x
    srelx = 0.04 # 4% of x
    cabsx = 0.03 # correlated x
    crelx = 0.02 # 2% of x correlated

    # perform fit to data with function mFit using class mnFit
    parvals, parerrs, cor, chi2 = mFit(fitmodel, data_x, data_y,
                                     sx=sabsx,
                                     sy=sabsy,
                                     srelx=srelx,
                                     srely=srely,
                                     xabscor=cabsx,
                                     xrelcor=crelx,
                                     yabscor=cabsy,
                                     yrelcor=crely,
  ##                                   p0=(1., 0.5),     
  #                                     constraints=['A', 1., 0.03],
  #                                     constraints=[0, 1., 0.03],
                                     use_negLogL=True,
                                     plot=True,
                                     plot_band=True,
                                     plot_cor=False,
                                     showplots=False,
                                     quiet=False,
                                     axis_labels=['x', 'y   \  f(x, *par)'], 
                                     data_legend = 'random data',    
                                     model_legend = 'model')
    plt.suptitle("mnFit example: fit to x-y data",
               size='xx-large', color='darkblue')

  # Print results 
    print('\n*==* xyFit Result:')
    print(" chi2: {:.3g}".format(chi2))
    print(" parameter values:      ", parvals)
    print(" neg. parameter errors: ", parerrs[:,0])
    print(" pos. parameter errors: ", parerrs[:,1])
    print(" correlations : \n", cor)


  def example_histogramFit():
  #
  # *** Histogram Fit: Example of an application of phyFit.hFit() 
  #

  #    # define the model function to fit
    def SplusB_model(x, mu = 6.0, sigma = 0.5, s = 0.3):
      '''pdf of a Gaussian signal on top of flat background
      '''
      normal = np.exp(-0.5*((x-mu)/sigma)**2)/np.sqrt(2.*np.pi*sigma**2)
      flat = 1./(xmx-xmn) 
      return s * normal + (1-s) * flat 

    nbins=40
    xmn = 1
    xmx = 10
    bedges=np.linspace(xmn, xmx, nbins+1)
    bcontents = np.array([1, 1, 1, 2, 2, 2, 6, 1, 0, 3, 1, 1, 0,
                        2, 3, 3, 1, 1, 0, 2, 3, 2, 3, 1, 1, 8,
                        6, 7, 9, 1, 0, 1, 2, 6, 3, 1, 3, 3, 3, 4])
    #  
    # ---  perform fit  
    #
    pvals, perrs, cor, gof = hFit(SplusB_model,
          bcontents, bedges,  # bin entries and bin edges
          p0=None,                # initial guess for parameter values 
     #   constraints=['s', val , err ],   # constraints within errors
          limits=('s', 0., None),  #limits
          use_GaussApprox=False,   # Gaussian approximation
          fit_density = True,      # fit density
          plot=True,           # plot data and model
          plot_band=True,      # plot model confidence-band
          plot_cor=False,      # plot profiles likelihood and contours
          showplots=False,      # show / don't show plots
          quiet=True,         # suppress informative printout
          axis_labels=['x', 'y   \  f(x, *par)'], 
          data_legend = 'random data',    
          model_legend = 'signal + background model' )

    plt.suptitle("mnFit example: fit to histogram data",
              size='xx-large', color='darkblue')

    # Print results 
    print('\n*==* histogram fit Result:')
    print(" goodness-of-fit: {:.3g}".format(gof))
    print(" parameter values:      ", pvals)
    print(" neg. parameter errors: ", perrs[:,0])
    print(" pos. parameter errors: ", perrs[:,1])
    print(" correlations : \n", cor)

  def likelihood_Fit():
    """**unbinned ML fit** with user-defined cost function

    This code illustrates uasge of the wrapper function userFit() 
    for  class **mnFit** 
    """  

    # generate Gaussian-distributed data
    mu0=2.
    sig0=0.5
    data = mu0 + sig0 * np.random.randn(100)

    # define cost function: 2 * negative log likelihood of Gauß;
    def myCost(mu=1., sigma=1.):
      r= (data-mu)/sigma
      return np.sum( r*r + 2.*np.log(sigma))

    pvals, perrs, cor, gof = userFit(myCost,
          p0=None,                 # initial guess for parameter values 
        #  limits=('sigma', None, None),  #limits
        #  constraints=['mu', 2., 0.01], # Gaussian parameter constraints
          neg2logL = True,         # cost ist -2 * ln(L)
          plot_cor=True,           # plot profiles likelihood and contours
          showplots=False,         # show / don't show plots
          quiet=True,              # suppress informative printout
          )

    plt.suptitle("Maximum-likelihood fit: profiles and contours",
                     size='xx-large', color='darkblue')
    # Print results
    print('\n*==* user-defined cost: Fit Result:')
    print(" parameter values:      ", pvals)
    print(" neg. parameter errors: ", perrs[:,0])
    print(" pos. parameter errors: ", perrs[:,1])
    print(" correlations : \n", cor)  

  #
  # --- run examples
  #
  example_xyFit()
  example_histogramFit()
  likelihood_Fit()

  # show all figures
  plt.show()
