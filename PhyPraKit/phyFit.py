"""package phyFit.py
  
  Physics Fitting with `iminiut` [https://iminuit.readthedocs.ios/en/stable/]

  Author: Guenter Quast, initial version Jan. 2021, updated Jun. 2021

  Requirements: 
   - Python >= 3.6
   - iminuit vers. > 2.0
   - scipy > 1.5.0
   - matplotlib > 3

  The class `mnFit.py` uses the optimization and uncertainty-estimation
  package `iminuit` for fitting a parameter-dependent model f(x, \*par) 
  to data points (x, y) or a probability density function to binned 
  histogram data or to unbinned data. Parameter estimation is based on 
  pre-implemented Maximum-Likelihood methods, or on a user-defined 
  cost function in the latter case, which provides maximum flexibility.
  Classical least-square methods are optionally available for comparison 
  with other packages. 

  A unique feature of the package is the support of different kinds 
  of uncertainties for x-y data, namely independent and/or correlated 
  absolute and/or relative uncertainties in the x and/or y directions. 
  Parameter estimation for density distributions is based on the shifted 
  Poisson distribution, Poisson(x - loc, lambda), of the number of entries 
  in each bin of a histogram. 

  Parameter constraints, i.e. external knowledge of parameters within
  Gaussian uncertainties, limits on parameters in order to avoid 
  problematic regions in parameter space during the minimization process, 
  and fixing of parameters, e.g. to include the validity range of a model
  in the parameters without affecting the fit, are also supported by *mnFit*.

  Method:
    Uncertainties that depend on model parameters are treated by dynamically 
    updating the cost function during the fitting process with `iminuit`.
    Data points with relative errors can thus be referred to the model 
    instead of the data. The derivative of the model function w.r.t. x is 
    used to project the covariance matrix of x-uncertainties on the y-axis. 

  Example functions *xyFit()*, *hFit()* and *mFit()*, illustrate how to 
  control the  interface of `mnFit`. A short example script is also 
  provided to perform fits on sample data. The sequence of steps performed
  by these interface functions is rather general and straight-forward:
  
  .. code-block:: python
  
     Fit = mnFit(fit_type)                # initialize a mnFit object 
     Fit.setOptions(run_minos=True, ...)  # set options
     Fit.init_data(data, parameters ...)  # initialize data container
     Fit.init_fit(ufcn, p0 = p0, ...)     # initialize Fit (and minuit)
     resultDict = Fit.do_fit()            # perform the fit (returns dictionary)
     resultTuple = Fit.getResult()        # retrieve results as tuple of np-arrays
     Fit.plotModel()                      # plot data and best-fit model
     Fit.plotContours()                   # plot profiles and confidence contours

  It is also possible to run a fit without the need to provide own
  Python code. In this case, the data, uncertainties and the model
  are read from a file in yaml format and passed to the function 
  xyFit_from_yaml() as a dictionary. 

  The implementation of the fitting procedure in this package is 
  - intentionally - rather minimalistic, and it is meant to 
  illustrate the principles of an advanced usage of `iminuit`. It 
  is also intended to stimulate own applications of special, 
  user-defined cost functions.

  The main features of this package are:
    - provisioning of cost functions for x-y and binned histogram fits 
    - implementation of the least-squares method for correlated Gaussian errors
    - support for correlated x-uncertainties by projection on the y-axis
    - support of relative errors with reference to the model values
    - shifted Poisson distribution for binned likelihood fits to histograms 
    - evaluation of profile likelihoods to determine asymmetric uncertainties
    - plotting of profile likelihood and confidence contours

  The **cost function** that is optimized for x-y fits basically is a 
  least-squares one, which is extended if parameter-dependent uncertainties 
  are present. In the latter case, the logarithm of the determinant of the 
  covariance matrix is added to the least-squares cost function, so that it 
  corresponds to twice the negative log-likelihood of a multivariate Gaussian 
  distribution. Fits to histogram data rely on the negative log-likelihood
  of the Poisson distribution, generalized to support fractional observed 
  values, which may occur if corrections to the observed bin counts have 
  to be applied. If there is a difference *DeltaMu* between the mean value
  and the variance of the number of entries in a bin due to corrections, 
  a "shifted Poisson distribution", Poiss(x-DeltaMu, lambda), is supported.

  Fully functional applications of the package are illustrated in executable
  script below, which contains sample data, executes the fitting procedure
  and collects and displays the results.

.. moduleauthor:: Guenter Quast <g.quast@kit.edu>
"""

import sys
import numpy as np, matplotlib.pyplot as plt
from scipy import stats, linalg
from scipy.special import loggamma
from scipy.optimize import newton
from inspect import signature
from iminuit import __version__ as iminuit_version
from iminuit import Minuit

def xyFit(fitf, x, y, sx = None, sy = None,
       srelx = None, srely = None, 
       xabscor = None, xrelcor = None,        
       yabscor = None, yrelcor = None,
       ref_to_model = True, 
       model_kwargs = None,
       p0 = None, dp0 = None, constraints = None,         
       fixPars=None, limits=None,
       use_negLogL=True, 
       plot = True, plot_cor = False,
       showplots = True, 
       plot_band=True, plot_residual=False, quiet = True,
       same_plot = False,    
       axis_labels=['x', 'y = f(x, *par)'],
       data_legend = 'data',    
       model_legend = 'model',
       return_fitObject=False ) :
  
  """Wrapper function to fit an arbitrary function fitf(x, \*par) 
  to data points (x, y) with independent and/or correlated absolute 
  and/or relative uncertainites on x- and/or y- values with class mnFit.

  Correlated absolute and/or relative uncertainties of input data 
  are specified as floats (if all uncertainties are equal) or as 
  numpy-arrays of floats. The concept of independent or common 
  uncertainties of (groups) of data points is used to construct the 
  full covariance matrix from different uncertainty components.
  Independent uncertainties enter only in the diagonal, while correlated 
  ones contribute to diagonal and off-diagonal elements of the covariance 
  matrix. Values of 0. may be specified for data points not affected by a 
  certain type of uncertainty. E.g. the array [0., 0., 0.5., 0.5] specifies
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
    * model_kwargs: optional, fit parameters if not from model signature 
    * p0: array-like, initial guess of parameters
    * dp0: array-like, initial guess of parameter uncertainties (optional)
    * use_negLogL:  use full -2ln(L)  
    * constraints: (nested) list(s) [name or id, value, error]
    * fix parameter(s) in fit: list of parameter names or indices
    * limits: (nested) list(s) [name or id, min, max] 
    * plot: show data and model if True
    * plot_cor: show profile likelihoods and confidence contours
    * plot_band: plot uncertainty band around model function
    * plot_residual: plot residuals w.r.t. model instead of model function
    * showplots: show plots on screen
    * same_plot: overlay fit info from previoius call(s)
    * quiet: suppress printout
    * list of str: axis labels
    * str: legend for data
    * str: legend for model 
    * bool: for experts only, return instance of class mnFit to 
      give access to data members and methods

  Returns:
    * np-array of float: parameter values
    * 2d np-array of float: parameter uncertainties [0]: neg. and [1]: pos. 
    * np-array: correlation matrix 
    * float: 2*negLog L, corresponding to \chi-square of fit a minimum

    or, optionally, the mnFit instance.
  """

  ## from .phyFit import mnFit #! already contained in this file

  # ... check if errors are provided ...
  if sy is None and srely is None:
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
   # pass model function, start parameter and possible constraints
  Fit.init_fit(fitf, model_kwargs=model_kwargs, p0=p0, dp0=dp0,
               constraints=constraints,
               fixPars=fixPars,
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
                 plot_band=plot_band,
                 same_plot = same_plot,
                 plot_residual=plot_residual)

  # figure with visual representation of covariances
  #   profile likelihood scan and confidence contours
  if plot_cor:
    fig_cor = Fit.plotContours(figname="xyFit: Profiles and Contours")

  # show plots on screen
  if showplots and (plot or plot_cor):
    plt.show()

  if return_fitObject:
    return Fit
  else:
    # return
    #   numpy arrays with fit result: parameter values,
    #   negative and positive parameter uncertainties,
    #   correlation matrix
    #   gof
    #   parameter names
    return Fit.getResult()

def xyFit_from_yaml(fd,                 # dictionary definig fit input
                    plot=True,          # plot data and model
                    plot_band=True,     # plot model confidence-band
                    plot_cor=False,     # plot profiles likelihood and contours
                    showplots = True,   # show plots on screen
                    same_plot = False,  # overlay fit info from previoius call(s)
                    quiet=True,         # suppress informative printout
                    return_fitObject=False # 
           ):

  """Perform fit with data and model from yaml file 

  Fit to x-y data with independent and correlated, absolute and relative 
  uncertainties in the x and y directions read from dictionary. The
  fitting procedure uses phyfit.xyFit().

  Args:
    * fd: fit input as a dictionary, extracted from a file in yaml format
    * plot: show data and model if True
    * plot_cor: show profile likelihoods and confidence contours
    * plot_band: plot uncertainty band around model function
    * plot_residual: plot residuals w.r.t. model instead of model function
    * showplots: show plots on screen - switch off if handled by calling process
    * quiet: suppress informative printout

  Returns:
    * result dictionary
    * optionally: produces result plots 

  **fit dictionary format:**

  .. code-block:: yaml

    label: <str data-set name>

    x_label: <str name x-data>
    x_data: [  list of float ]   

    y_label: <str name y-data>  
    y_data: [ list of float ]

    x_errors: <float>, [list of floats], or {dictionary/ies}
    y_errors:  <float>, [list of floats], or {dictionary/ies}

    model_label: <str model name>
    model_function: |
      <Python code>

    format of uncertainty dictionary: 
    - error_value: <float> or [list of floats]
    - correlation_coefficient: 0. or 1.
    - relative: true or false
    relative errors may be spcified as <float>%

    fix_parameters:
      - <name1> 
      - <name2>
        ...
     
    parameter_constraints:
      <name1>:
        value: <v>
        uncertainty <u>
      <name2>:
        ...

  simple example of *yaml* input:

  .. code-block:: yaml

    label: 'Test Data'

    x_data: [.05,0.36,0.68,0.80,1.09,1.46,1.71,1.83,2.44,2.09,3.72,4.36,4.60]
    x_errors: 3%
    x_label: 'x values'

    y_data: [0.35,0.26,0.52,0.44,0.48,0.55,0.66,0.48,0.75,0.70,0.75,0.80,0.90]
    y_errors: [.06,.07,.05,.05,.07,.07,.09,.1,.11,.1,.11,.12,.1]
    y_label: 'y values'

    model_label: 'Parabolic Fit'
    model_function: |
      def quadratic_model(x, a=0., b=1., c=0. ):
        return a * x*x + b*x + c

  .. moduleauthor:: Guenter Quast <g.quast@kit.edu>
  """

  from PhyPraKit import check_function_code
  ## from .phyFit import xyFit #! already contained in this file

  # -- define some helper fuctions
  def decode_uDict(uDict):
    """Decode dictionary with uncertainties

      yaml format:
      y-errors:
      - type:                     "simple" or "matrix"
        error_value:              number or array
        correlation_coefficient:  0. or 1.
        relative:                 false or true

      Args:
        - uDict : dictionary with uncertainties
     
      Returns:  
        - s       # independent errors
        - srel    # independent relative errors
        - sabscor # correlated absolute errors
        - srelcor # correlated relative errors
    """
  
    def decode_rel(e):
      """Decode numbers with %-sign: interpreted as relative errors
      """
      if type(e) is type(''):
        if '%' in e:
          v = float(e[0:e.find('%')])/100.
      else:
          v = e
      return v

    s = None        # independent errors
    srel = None     # independent relative errors
    sabscor = None  # correlated absolute errors
    srelcor = None  # correlated relative errors

    if type(uDict) is not type([]):      # got a scalar, one error for all
      if type(uDict) is type(''):
        if '%' in uDict:
          srel=decode_rel(uDict)
      else:
        s = uDict
      return s, srel, sabscor, srelcor

    if type(uDict[0]) is not type({}): # array of uncertainties, one for each data point
      for i, v in enumerate(uDict):
        if type(v) is type(''):
          uDict[i]=decode_rel(v) 
          srel= uDict
        else:                          
          s = uDict
      return s, srel, sabscor, srelcor
        
    # else decode complex error dictionary
    for ed in uDict:
      if 'type' in ed:
        typ = ed['type']
      else:
        typ = "simple"
      if 'relative' in ed:
        rel = ed['relative']
      else:
        rel = False   
      if 'correlation_coefficient' in ed:
        cor = ed['correlation_coefficient'] 
      else:
        cor = 0.
      if 'error_value' in ed:
        e = ed['error_value']
        if type(e) is type([]):
          for i,v in enumerate(e):
            if type(v) is type(''):  
              if '%' in v:
                rel = True
                e[i]=decode_rel(v)
        else:
          if type(e) is type(''):  
            if '%' in e:
              rel = True
              e=decode_rel(e)          
      else:
        e = None
     #    
      if cor == 0. and not rel:
        # independent absolute error
        if s is None:
          s = e
        else:
          s = [s]
          s.append(e)
      elif cor == 0. and rel:
        # independent relative error
        if srel is None:
          srel = e
        else:
          srel = [srel]
          srel.append(e)
      elif cor !=0 and not rel:
        # correlated absolute error
        if sabscor is None:
          sabscor = e
        else:
          sabscor = [sabscor]
          sabscor.append(e)
      elif cor !=0 and rel:
        # correlated relative error
        if srelcor is None:
          srelcor = e
        else:
          srelcor = [srelcor]
          srelcor.append(e)
     # -- end for 
    return s, srel, sabscor, srelcor  

# --- end helper functions 

  # Extract information from input dictionary   
  if 'label' in fd:
    data_label = fd['label']
  else:
    data_label = 'data'

# - data
  data_x = list(map(float, fd['x_data']))
  data_y = list(map(float, fd['y_data']))
  if 'x_label' in fd:
    x_label = fd['x_label']
  else:
    x_label = 'x'
  if 'y_label' in fd:
    y_label = fd['y_label']
  else:
    y_label = 'y'
    
# - uncertainties
#   x errors
  if 'x_errors' in fd:
    sx, srelx, sabscorx, srelcorx = decode_uDict(fd['x_errors'])
  else:
    sx = None
    srelx = None
    sabscorx = None
    srelcorx = None
  
#  y errors 
  try:      # look at to two different places
    uDict = fd['parametric_model']['y_errors']
  except:
    try:
      uDict = fd['y_errors']
    except:
      raise ValueError("!!! no y-errors found !") # must have y uncertainties 
  sy, srely, sabscory, srelcory = decode_uDict(uDict)

# fix paramerters
  fixedPars = None
  if 'fix_parameters' in fd:
    fixedPars = []
    for pnam in fd['fix_parameters']:
      fixedPars.append(pnam)
  
# constraints
  constraints = None
  if 'parameter_constraints' in fd:
    constraints =  []
    for pnam in fd['parameter_constraints']:
      vu_dict = fd['parameter_constraints'][pnam]
      v = vu_dict['value']
      u = vu_dict['uncertainty']
      constraints.append([pnam, v, u ])
      
# - model name and python code
  if 'model_label' in fd:
    model_label = fd['model_label']
  else:
    model_label = 'model'
  try:
    code_str = fd['model_function']['python_code']
  except:
    try:
      code_str = fd['model_function']
    except:
      raise ValueError("!!! no code to fit found !") 
  fitf_name, code = check_function_code(code_str)    

# print input and model
  if not quiet:
    print('\n*==* xyFit_from_yaml:')
    print('-- input data:')
    print('x-data:', data_x)
    print('+/- abs', sx)
    print('    rel', srelx)
    print('   cabs', sabscorx)
    print('   crel', srelcorx)
    print('y-data:', data_y)
    print('+/- abs', sy)
    print('    rel', srely)
    print('   cabs', sabscory)
    print('   crel', srelcory)
    if constraints is not None:
      print('  paramter constraints:', constraints)
    print('-- model to fit: %s' %(fitf_name))
    print('%s' %(code) )    

  scope = dict()
  header= 'import numpy as np\n' + 'import scipy\n' 
  exec(header + code, scope)
  
# perform fit to data with function xyFit 
  rdict = xyFit(scope[fitf_name],
      data_x, data_y,      # data x and y coordinates
      sx=sx,               # indep x
      sy=sy,               # indep y
      srelx=srelx,         # indep. rel. x
      srely=srely,         # indep. rel. y
      xabscor=sabscorx,    # correlated x
      xrelcor=srelcorx,    # correlated rel. x
      yabscor=sabscory,    # correlated y
      yrelcor=srelcory,    # correlated rel. y
      constraints = constraints, # parameter constraints
      fixPars = fixedPars, # fixed parameters
      axis_labels=[x_label, y_label], 
      data_legend =  data_label,    
      model_legend = model_label, 
      plot=plot,             # plot data and model
      plot_band=plot_band,   # plot model confidence-band
      plot_cor=plot_cor,     # plot profiles likelihood and contours
      showplots=showplots,   # plots to screen
      same_plot = same_plot, # overlay fit info from previoius call(s)
      quiet=quiet,           # suppress informative printout
      return_fitObject=return_fitObject
      )
  return rdict
   
def xFit(fitf, x, s = None, srel = None, 
       sabscor = None, srelcor = None,               
       ref_to_model = True,
       names = None,
       model_kwargs = None,
       p0 = None, dp0 = None, constraints = None,
       fixPars=None, limits=None,
       use_negLogL=True, 
       plot = True, plot_cor = False,
       showplots = True, 
       plot_band=True, plot_residual=False, quiet = True,
       same_plot = False,    # overlay fit info from previoius call(s)
       axis_labels=['Index', 'f(*x, *par)'],
       data_legend = 'data',    
       model_legend = 'model',
       return_fitObject = False ):
  
  """Wrapper function to fit an arbitrary function to data with
  independent and/or correlated absolute and/or relative uncertainties 
  with class mnFit. Uncertainties are assumed to be desribed by
  a multivariate Gaussian distribution, i.e. the covariance matrix of 
  the data {x_i} is taken into account in the cost function.

  Correlated absolute and/or relative uncertainties of input data 
  are specified as floats (if all uncertainties are equal) or as 
  numpy-arrays of floats. The concept of independent or common 
  uncertainties of (groups) of values is used to construct the 
  full covariance matrix from different uncertainty components.

  Args:
    * fitf: model function to fit, arguments (float:x, float: \*args)
    * x:  np-array of data values
    * s: scalar or 1d or 2d np-array with uncertainties
    * srel: scalar or np-array; relative uncertainties 
    * sabscor: scalar or np-array; absolute, correlated error(s) 
    * srelcor: scalar or np-array; relative, correlated error(s) 
    * ref_to_model: relative errors w.r.t. model if True
    * names: optional names for each input value
    * model_kwargs: optional, fit parameters if not from model signature 
    * p0: array-like, initial guess of parameters
    * dp0: array-like, initial guess of parameter uncertainties (optional)
    * use_negLogL:  use full -2ln(L)  
    * constraints: (nested) list(s) [name or id, value, error]
    * fix parameter(s) in fit: list of parameter names or indices
    * limits: (nested) list(s) [name or id, min, max] 
    * plot: show data points and model prediction if True
    * plot_cor: show profile likelihoods and confidence contours
    * plot_band: plot uncertainty band around model prediction
    * plot_residual: plot residuals w.r.t. model instead of model prediction
    * showplots: show plots on screen
    * same_plot: overlay fit info from previous call(s)
    * quiet: suppress printout
    * list of str: axis labels
    * str: legend for data
    * str: legend for model 
    * bool: for experts only, return instance of class mnFit to 
      give access to data members and methods

  Returns:
    * np-array of float: parameter values
    * 2d np-array of float: parameter uncertainties [0]: neg. and [1]: pos. 
    * np-array: correlation matrix 
    * float: 2*negLog L, corresponding to \chi-square of fit a minimum

    or, optionally, the mnFit instance.
  """

  ## from .phyFit import mnFit #! already contained in this file

  # ... check if errors are provided ...
  if s is None:
    sy = np.ones(len(x))
    print('\n!**! No y-errors given, all assumed to be 1.0\n',
          '-> consider scaling of parameter errors with sqrt(chi^2/Ndf)\n')
  
  # set up a fit object
  indexedFit = mnFit('indexed')

  # set some options
  indexedFit.setOptions(run_minos=True,
                  relative_refers_to_model=ref_to_model,
                  use_negLogL=use_negLogL,
                  quiet=quiet)

  # pass data and uncertainties to fit object
  indexedFit.init_data(x, e = s, 
                       erel = srel, cabs = sabscor, crel = srelcor,
                       names = names)
   # pass model function, start parameter and possible constraints
  indexedFit.init_fit(fitf, model_kwargs=model_kwargs, p0=p0, dp0=dp0, 
               constraints=constraints,
               fixPars=fixPars,
               limits=limits)
   # perform the fit
  fitResult = indexedFit.do_fit()
  # print fit result(dictionary from migrad/minos(
  if not quiet:
    print("\nFit Result from migrad:")
    print(fitResult[0])
    if fitResult[1] is not None:
      print("\nResult of minos error analysis:")
      print(fitResult[1])
    
  # produce figure with data and model
  if plot:
    fig = indexedFit.plotModel(axis_labels=axis_labels,
                 data_legend=data_legend,
                 model_legend=model_legend,
                 plot_band=plot_band,
                 same_plot = same_plot,
                 plot_residual=plot_residual)

  # figure with visual representation of covariances
  #   profile likelihood scan and confidence contours
  if plot_cor:
    fig_cor = indexedFit.plotContours(figname="xyFit: Profiles and Contours")

  # show plots on screen
  if showplots and (plot or plot_cor):
    plt.show()

  if return_fitObject:
    return indexedFit
  else:
    # return
    #   numpy arrays with fit result: parameter values,
    #   negative and positive parameter uncertainties,
    #   correlation matrix
    #   gof
    #   parameter names
    return indexedFit.getResult()

def hFit(fitf, bin_contents, bin_edges, DeltaMu=None,
         model_kwargs = None,
         p0 = None, dp0 = None, constraints = None,
         fixPars=None, limits=None,
         use_GaussApprox = False,
         fit_density = True,
         plot = True, plot_cor = False,
         showplots = True, plot_band=True, plot_residual=False,
         same_plot = False,    # overlay fit info from previoius call(s)
         quiet = True,
         axis_labels=['x', 'counts/bin = f(x, *par)'],
         data_legend = 'Histogram Data',    
         model_legend = 'Model',
         return_fitObject=False ) :
  
  """Wrapper function to fit a density distribution f(x, \*par) 
  to binned data (histogram) with class mnFit 
  
  The cost function is two times the negative log-likelihood of the Poisson 
  distribution, or - optionally - of the Gaussian approximation.

  Uncertainties are determined from the model values in order to avoid biases 
  and to take account of empty bins of an histogram. The default behaviour is 
  to fit a normalised density; optionally, it is also possible to fit the 
  number of bin entries.

  Args:
    * fitf: model function to fit, arguments (float:x, float: \*args)
    * bin_contents:  
    * bin_edges: 
    * DeltaMu: shift mean (=mu) vs. variance (=lam), for Poisson: mu=lam
    * model_kwargs: optional, fit parameters if not from model signature 
    * p0: array-like, initial guess of parameters
    * dp0: array-like, initial guess of parameter uncertainties (optional)
    * constraints: (nested) list(s) [name or id, value, error] 
    * limits: (nested) list(s) [name or id, min, max] 
    * use_GaussApprox: Gaussian approximation instead of Poisson 
    * density: fit density (not number of events)
    * plot: show data and model if True
    * plot_cor: show profile likelihoods and confidence contours
    * plot_band: plot uncertainty band around model function
    * same_plot: overlay fit info from previous call(s)
    * plot_residual: plot residuals w.r.t. model instead of model function
    * showplots: show plots on screen
    * quiet: suppress printout
    * axis_labes: list of tow strings, axis labels
    * data_legend: legend entry for data
    * model_legend: legend entry for model 
    * bool: for experts only, return instance of class mnFit to give access 
      to data members and methods

  Returns:
    * np-array of float: parameter values
    * 2d np-array of float: parameter uncertainties [0]: neg. and [1]: pos. 
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
  Fit.init_fit(fitf, model_kwargs=model_kwargs, p0=p0, dp0=dp0,
               constraints=constraints,
               fixPars=fixPars,
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
                 plot_band=plot_band,
                 same_plot = same_plot,
                 plot_residual=plot_residual)

  # figure with visual representation of covariances
  #   profile likelihood scan and confidence contours
  if plot_cor:
    fig_cor = Fit.plotContours(figname="histFit: Profiles and Contours")

  # show plots on screen
  if showplots and (plot or plot_cor):
    plt.show()

  if return_fitObject:
    return Fit
  else:
    # return
    #   numpy arrays with fit result: parameter values,
    #   negative and positive parameter uncertainties,
    #   correlation matrix
    #   gof
    #   parameter names
    return Fit.getResult()

def hFit_from_yaml(fd,           # dictionary defining fit input
             plot=True,          # plot data and model
             plot_band=True,     # plot model confidence-band
             plot_cor=False,     # plot profiles likelihood and contours
             showplots = True,   # show plots on screen
             same_plot = True,   # overlay fit info from previoius call(s)
             quiet=True,         # suppress informative printout
             return_fitObject=False # 
           ):

  """Binned log-likelihood fit to histogram data with input from yaml file with phyfit.hFit()

  Args:
    * fd: fit input as a dictionary, extracted from a file in yaml format
    * plot: show data and model if True
    * plot_cor: show profile likelihoods and confidence contours
    * plot_band: plot uncertainty band around model function
    * plot_residual: plot residuals w.r.t. model instead of model function
    * showplots: show plots on screen - switch off if handled by calling process
    * quiet: suppress informative printout

  Returns:
    * result dictionary
    * optionally: produces result plots 

  simple example of *yaml* input:

  .. code-block:: yaml

    # Example of a fit to histogram data
    type: histogram

    label: example data
    x_label: 'h' 
    y_label: 'pdf(h)'

    # data:
    raw_data: [ 79.83,79.63,79.68,79.82,80.81,79.97,79.68,80.32,79.69,79.18,
            80.04,79.80,79.98,80.15,79.77,80.30,80.18,80.25,79.88,80.02 ]

    n_bins: 15
    bin_range: [79., 81.]
    # alternatively an array for the bin edges can be specified
    #bin_edges: [79., 79.5, 80, 80.5, 81.]

    model_density_function: |
      def normal_distribution(x, mu=80., sigma=1.):
        return np.exp(-0.5*((x - mu)/sigma)** 2)/np.sqrt(2.*np.pi*sigma** 2)

  .. moduleauthor:: Guenter Quast <g.quast@kit.edu>
  """

  from PhyPraKit import check_function_code
  ## from .phyFit import hFit #! already contained in this file

  # Extract information from input dictionary   
  if 'label' in fd:
    data_label = fd['label']
  else:
    data_label = 'data'

# - data
  hdata = list(map(float, fd['raw_data']))
  bins = 10
  if 'n_bins' in fd:
    bins = fd['n_bins']
  if 'bin_edges' in fd:
    bins = fd['bin_edges']
  bin_range = None
  if 'bin_range' in fd:
    bin_range = fd['bin_range']

  if 'x_label' in fd:
    x_label = fd['x_label']
  else:
    x_label = 'x'
  if 'y_label' in fd:
    y_label = fd['y_label']
  else:
    y_label = 'y'

# fix paramerters
  fixedPars = None
  if 'fix_parameters' in fd:
    fixedPars = []
    for pnam in fd['fix_parameters']:
      fixedPars.append(pnam)
  
# constraints
  constraints = None
  if 'parameter_constraints' in fd:
    constraints =  []
    for pnam in fd['parameter_constraints']:
      vu_dict = fd['parameter_constraints'][pnam]
      v = vu_dict['value']
      u = vu_dict['uncertainty']
      constraints.append([pnam, v, u ])
        
# - model name and python code
  if 'model_label' in fd:
    model_label = fd['model_label']
  else:
    model_label = 'model'
  try:
    code_str = fd['model_density_function']['python_code']
  except:
    try:
      code_str = fd['model_density_function']
    except:
      try:
        code_str = fd['model_function']
      except:
        raise ValueError("!!! no code to fit found !") 
  fitf_name, code = check_function_code(code_str)    

  bin_contents, bin_edges = np.histogram(hdata, bins=bins,  range=bin_range)
  
  # execute definition of model function
  scope = dict()
  header= 'import numpy as np\n' + 'import scipy\n' 
  exec(header + code, scope)

  # perform fit to data with function hFit 
  rdict = hFit(scope[fitf_name],
         bin_contents, bin_edges,
         fixPars = fixedPars,
         constraints = constraints,      
         plot = plot,
         plot_cor = plot_cor,
         plot_band=plot_band,
         same_plot = same_plot,
         quiet = quiet,
         showplots = showplots,
         axis_labels=[x_label, y_label],
         data_legend = data_label,    
         model_legend = model_label,
         return_fitObject=return_fitObject
         ) 
  return rdict

  
def mFit(ufcn, data = None,
         model_kwargs = None,
         p0 = None, dp0 = None, 
         constraints = None, limits=None, fixPars=None,
         neg2logL = True,
         plot = False, plot_band = True,
         plot_cor = False,
         showplots = True, quiet = True,
         same_plot = False,
         axis_labels=['x', 'Density = f(x, *par)'],
         data_legend = 'data',    
         model_legend = 'model',
         return_fitObject=False ) :

  """Wrapper function to directly fit a user-defined cost funtion

  This is the simplest fit possible with the class mnFit. If no data is
  specified (data=None), a user-supplied cost function (ufcn) is minimized 
  and an estimation of the parameter uncertainties performed, assuming
  the cost function is a negative log-likelihood function (nlL of 2nLL).
  
  In case data is provided, the user function `ufcn(data, *par)` is 
  interpreted as a parameter-dependent probability density function, and
  the parameters are determined in an unbinned log-likelihood approach. 

  Args:
    * ufcn: user-defined cost function or pdf to be minimized; 

      - ufcn(\*par): the uncertainty estimation relies on this being a 
        negative log-likelihood function ('nlL'); in this case, no data
        is to be provided, i.e. `data=None`. 

      - ufcn(x, \*par): a probability density of the data `x` depending on
        the set of parameters `par`. 

    * data, optional, array of floats: optional input data
    * model_kwargs: optional, fit parameters if not from model signature 
    * p0: array-like, initial guess of parameters
    * dp0: array-like, initial guess of parameter uncertainties (optional)
    * constraints: (nested) list(s) [name or id, value, error] 
    * limits: (nested) list(s) [name or id, min, max] 
    * neg2logL: use 2 * nlL (corresponding to a least-squares-type cost)
    * plot: show data and model if True
    * plot_band: plot uncertainty band around model function
    * plot_cor: plot likelihood profiles and confidence contours of parameters
    * showplots: show plots on screen (can also be done by calling script)
    * quiet: controls verbose output
    * bool: for experts only, return instance of class mnFit to 
      give access to data members and methods
    """

  # from .phyFit import mnFit # already contained in phyFit package

  if data is None:
    fit_type="user"
  else:
    fit_type = "ml"
  uFit = mnFit(fit_type)

  # set options
  uFit.setOptions(run_minos = True,
                 neg2logL = neg2logL,
                 quiet = quiet)

  # initialize data container if data is provided
  #  !!! if no data given, the user-supplied cost function must
  #      handle got calulation of the cost function 
  if data is not None:
    uFit.init_data(data)

  # initialze fit
  #  - with the user-supplied cost function cost(*pars) if no data given
  #  - with probability density pdf(x; *pars) if data (=x) provided 
  uFit.init_fit(ufcn, model_kwargs=model_kwargs, p0=p0, dp0=dp0, 
                constraints = constraints,
                fixPars = fixPars,
                limits = limits)
  # perform the fit
  fitResult = uFit.do_fit()
  # print fit result (dictionary from migrad/minos(
  if not quiet:
    print("\nFit Result from migrad:")
    print(fitResult[0])
    if fitResult[1] is not None:
      print("\nResult of minos error analysis:")
      print(fitResult[1])
    
  # produce figure with data and model
  if plot:
    fig = uFit.plotModel(axis_labels=axis_labels,
                 data_legend=data_legend,
                 model_legend=model_legend,
                 plot_band=plot_band,
                 same_plot=same_plot)
  # figure with visual representation of covariances
  #    profile likelihood scan and confidence contours
  if plot_cor:
    fig_cor = uFit.plotContours(figname="userFit: Profiles and Contours")

  # show plots on screen
  if showplots and (plot or plot_cor):
    plt.show()

  if return_fitObject:
    return uFit
  else:
    # return
    #   numpy arrays with fit result: parameter values,
    #   negative and positive parameter uncertainties,
    #   correlation matrix
    #   gof
    #   parameter names
    return uFit.getResult()

#
# --- helper functions
#
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

def round_to_error(val, err, nsd_e=2):
  """round float *val* to same number of significant digits as uncertainty *err*
  
  Returns:
    * int:   number of significant digits for v
    * float: val rounded to precision of err
    * float: err rounded to precision nsd_e
  """
  v = abs(val)
  # round uncertainty to nd0 significant digits
  e = float("{:.{p}g}".format(abs(err), p=nsd_e))
  _v = v if v>e else e
  l10e=np.floor(np.log10(e))
  # determine # of significant digits for v
  _nd = int( np.floor(np.log10(_v) - l10e ) ) + nsd_e
  # take into account possible rounding of v ...
  v = float("{:.{p}g}".format(v, p=_nd))
  # ... and determine final # of sig. digits
  _v = v if v>e else e
  nsd_v = int( np.floor(np.log10(_v) - l10e ) ) + nsd_e
  v = float("{:.{p}g}".format(v, p=nsd_v)) if v>=10**(l10e-1) else 0
  return nsd_v, np.sign(val)*v, e               


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

#
# --- classes and functions
#

class mnFit():
  """**Fit an arbitrary function f(x, *par) to data**  
  with independent and/or correlated absolute and/or relative uncertainties

  This implementation depends on and heavily uses features of 
  the minimizer and uncertainty-estimator **iminuit**.
   
  Public Data member

  - fit_type: 'xy' (default), 'hist', 'user' or 'ml', controls type of fit 

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
  - plot_clContour():   plot confidence level contour for pair of parameters  
  - plot_nsigContour(): plot n-sigma contours for pair of parameters  
  - getProfile():       return profile likelihood of parameter pnam
  - getContour():       return contour points of pair of parameters

  Sub-Classes:

  - xyDataContainer:    Data and uncertainties for x-y data
  - histDataContainer:  Container for histogram data
  - mlDataContainter:   Container for general (indexed) data

  - xLSqCost:           Extended chi^2 cost function for fits to x-y data
  - hCost:              Cost function for (binned) histogram data 
    (2*negl. log. Likelihood of Poisson distribution)
  - mnCost:             user-supplied cost function or negative log-likelihood 
    of user-supplied probability distribution

  Methods:

  - init_xyData():       initialize xy data and uncertainties
  - init_hData():        initialize histogram data and uncertainties
  - init_mlData():       store data for unbinned likelihood-fit
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
  
  - iminuit_version     version of iminuit 
  - options, dict:      list of options 
  - ParameterNames:     names of parameters (as specified in model function)
  - nconstraints        number of constrained parameters    
  - nfixed              number of fixed parameters
  - freeParNams:        names of free parameters 
  - GoF:                goodness-of-fit, i.e. chi2 at best-fit point
  - NDoF:               number of degrees of freedom
  - ParameterValues:    parameter values at best-fit point
  - MigradErrors:       symmetric uncertainties
  - CovarianceMatrix:   covariance matrix
  - CorrelationMatrix:  correlation matrix
  - OneSigInterval:     one-sigma (68% CL) ranges of parameter values from MINOS  - ResultDictionary:   dictionary wih summary of fit results
 
  - for xyFit:
  
    - covx:     covariance matrix of x-data
    - covy:     covariance matrix of y-data 
    - cov:      combined covariance matrix, including projected x-uncertainties

  Instances of (sub-)classes:

  - minuit.\*: methods and members of Minuit object 
  - data.\*:   methods and members of data sub-class, generic
  - costf.\*:  methods and members of cost sub-class, generic 
  """

  def __init__(self, fit_type='xy'):
    """
    Type of fit:

    - 'xy'   : fit model model y(f(x; par) to data 
    - 'hist' : fit densitiy to binned data (i.e. histogram) 
    - 'user': user-supplied cost-function (i.e. neg. log-likelihood)
    """

    if fit_type not in ['xy', 'indexed', 'hist', 'user', 'ml']:
      raise ValueError(
        '!**! mnFit: invalid fit type ' + fit_type + ' - exiting!') 
    self.fit_type = fit_type

    self.iminuit_version = str(iminuit_version)

    # counter for number of external constraints 
    self.nconstraints = 0
    # counter for number of fixed parameters
    self.nfixed = 0 

    # set default of all options
    #
    #   no data or model provided yet
    self.xyData = None
    self.hData = None
    self.mlData = None
    self.xData = None
    # generic, holds active instance of sub-class xxData
    self.data = None
    # generic, holds active instanc of sub-class xxCost
    self.costf = None
    # no fit done yet
    self.migradResult = None
    self.minosResult = None
    self.migrad_ok = False
    self.minos_ok = False

    self.ResultDictionary = None
    
    # default options
    self.run_minos = True
    self.quiet = True
    # for xy and indexed Fit
    self.refModel=True
    self.use_negLogL = True
    self.iterateFit = False
    # for histogram fit
    self.use_GaussApprox = False
    self.fit_density = True
    # for fit with external cost
    self.neg2logL = True
    self.ErrDef = 1.
    # legend for possible options
    self.options = {}
    self.options["run_minos"] = [1, "all",
                  "no likelihood scan",
                  "MINOS profile likelihood scan"]
    self.options["refModel"]=[1, ["xy", "indexed"],
                            "relative uncertainties refer to data",
                            "relative uncertainties refer to model"]
    self.options["use_negLogL"] = [1, ["xy", "indexed"],
                                 "using simple chi^2 cost-function",
                                 "using full negative log-likelihood"]

    self.options["use_GaussApprox"] = [0, "hist",
            "using Poisson likelihood",
            "using Gaussian approximation of Poisson distibution"]
    self.options["fit_density"] = [1, "hist",
                                 "fit for number of entries/bin",
                                 "fit density distribution"]
    self.options["neg2logL"] = [1, ["user", "ml"],
                              "using standard likelihood -> errdef = 0.5",
                              "using -2 * neg. log. likelihood -> errdef=1."]

    # set options for (nicer) plotting
    self.setPlotOptions()
    
  def init_data(self, *args, **kwargs):
    if self.fit_type == 'xy':
      self.init_xyData(*args, **kwargs)
    elif self.fit_type == 'indexed':
      self.init_xData(*args, **kwargs)
    elif self.fit_type == 'hist':
      self.init_hData(*args, **kwargs)
    elif self.fit_type == 'ml':
      self.init_mlData(*args, **kwargs)
    elif self.fit_type == 'user':
      print("!**! mnFit: not data object to be definded for fit_type 'user'" )
    else:
      raise ValueError(
        "!**! unknown type of Fit " + self.fit_type)
    
  def setOptions(self, *args, **kwargs):
    if self.fit_type == 'xy':
      self.set_xyOptions(*args, **kwargs)
    elif self.fit_type == 'indexed':
      self.set_xOptions(*args, **kwargs)
    elif self.fit_type == 'hist':
      self.set_hOptions(*args, **kwargs)
    elif self.fit_type == 'user' or self.fit_type == 'ml':
      self.set_mnOptions(*args, **kwargs)
    else:
      raise ValueError(
        "!**! unknown type of Fit " + self.fit_type)

  def init_fit(self, *args, **kwargs):
    if self.fit_type == 'xy':
      self.init_xyFit(*args, **kwargs)
    elif self.fit_type == 'indexed':
      self.init_xFit(*args, **kwargs)
    elif self.fit_type == 'hist':
      self.init_hFit(*args, **kwargs)
    elif self.fit_type == 'user' or self.fit_type == 'ml':
      self.init_mnFit(*args, **kwargs)
    else:
      raise ValueError(
        "!**! unknown type of Fit " + self.fit_type)

  #
  # --- special code for xy Fit
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
      self.options["refModel"][0] = int(relative_refers_to_model)
    if run_minos is not None:   
      self.run_minos = run_minos
      self.options["run_minos"][0] = int(run_minos)
    if use_negLogL is not None:   
      self.use_negLogL = use_negLogL
      self.options["use_negLogL"][0] = int(use_negLogL)
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
    self.xyData = self.xyDataContainer(self, x, y, ex, ey,
                    erelx, erely, cabsx, crelx, cabsy, crely,
                    quiet=self.quiet)
    self.data = self.xyData
    # set flags for steering of fit process in do_fit()
    self.iterateFit = self.xyData.has_xErrors or(
         self.xyData.has_rel_yErrors and self.refModel)


  def init_xyFit(self, model, model_kwargs=None, p0=None, dp0=None, 
                 constraints=None,
                 fixPars=None,
                 limits=None):
    """initialize fit object

    Args:
      - model: model function f(x; \*par)
      - model_kwargs: optional, fit parameters if not from model signature 
      - p0: np-array of floats, initial parameter values 
      - dp0: array-like, initial guess of parameter uncertainties (optional)
      - constraints: (nested) list(s): [parameter name, value, uncertainty] 
        or [parameter index, value, uncertainty]
      - limits: (nested) list(s): [parameter name, min, max] 
        or [parameter index, min, max]
    """

    if self.xyData is None:
      raise ValueError(
        ' !!! mnFit.init_xyFit: no data object defined - call init_data()')

    # get parameters of model function to set start values for fit
    if model_kwargs is None:
      args, model_kwargs = get_functionSignature(model)

    par = (model_kwargs, p0, dp0, constraints, fixPars, limits)
    self._setupFitParameters(*par) 

    # create cost function
    self.costf = self.xLSqCost(self,
                           model,
                           use_neg2logL= self.use_negLogL)
    self._setupMinuit(model_kwargs) 

      
  class xyDataContainer:
    """
    Handle data and uncertainties and  
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
      - init_dynamicErrors():
      - get_Cov(): final covariance matrix (incl. proj. x)  
      - get_xCov(): covariance of x-values
      - get_yCov(): covariance of y-values
      - get_iCov(): inverse covariance matrix
      - plot(): provide a figure with representation of data
 
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
      # save poiner to calling class
      self.outer = outer

      # assume simple case w.o. cov.mat.
      self.needs_covariance = False 

      nd = len(x)      
      # store input data as numpy float arrays, ensure length nd if needed
      self.dlength = nd
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
      self.model_values = None

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
        err2 = _build_CovMat(self.nd,
                    self.ey, self.erely, self.cabsy, self.crely, self.y)
      else:
        err2 = _build_Err2(self.ey, self.erely, self.y)

      # initialize uncertainties and covariance matrix,
      self._initialCov(err2)
      # sets: 
      #   self.covx: covariance matrix of x
      #   self.covy: covariance matrix of y uncertainties
      #   self.cov: full covariance matrix incl. projected x
      #   self.iCov: inverse of covariance matrix
      #   self.err2: array of squared uncertainties
      #   self.iErr2: 1./self.err2
      
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
      
    def init_dynamicErrors(self):
      # method to switch on dynamic re-calculation of covariance matrix 
      self.ref_toModel = self.outer.refModel
      self.model = self.outer.costf.model
     
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
          self._staticCov = _build_CovMat(self.nd,
                       self.ey, eabscor = self.cabsy)
        else: 
          # all y-errors are parameter-independent
          self._staticCov = _build_CovMat(self.nd,
                              self.ey, erel=self.erely,
                              eabscor = self.cabsy, erelcor=self.crely,
                                            data=self.y)
        # build matrix of relative errors
        if self.ref_toModel and self.has_rel_yErrors:
          self._covy0 = _build_CovMat(self.nd,
                             erel=self.erely,
                             erelcor=self.crely,
                             data=np.ones(self.nd))
        else:
          self._covy0 = None
        # covariance matrix of x-uncertainties (all are parameter-dependent)
        if self.has_xErrors:
          self.covx = _build_CovMat(self.nd,
                         self.ex, self.erelx,
                         self.cabsx, self.crelx,
                         self.x)
         #  determine dx for derivative from smallest x-uncertainty
          self._dx = np.sqrt(min(np.diagonal(self.covx)))/10.
        else:
          self.covx = None

      else: # no covariance needed, use simple math
        # build static (=parameter-independent) part of covariance matrix      
        if self.has_rel_yErrors and self.ref_toModel:
          # only independent y-errors do not depend on parameters
          self._staticErr2 = _build_Err2(self.ey)
        else: 
          # all y-errors are parameter-independent
          self._staticErr2 = _build_Err2( self.ey, self.erely, self.y)
          
        if self.has_xErrors:
          self.err2x = _build_Err2(self.ex, self.erelx, self.x)
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
                   data_label='data',
                   plot_residual=False):
      """return figure with xy data and uncertainties
      """
#    # get data
      x = self.x
      if plot_residual and self.model_values is not None:
        y = self.y - self.model_values
      else:  
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
  class xLSqCost:
    """
    Custom e_x_tended Least-SQuares cost function with 
    dynamically updated covariance matrix and -2log(L) 
    correction term for parameter-dependent uncertainties.

    The default cost function is twice the negative logarithm 
    of the likelihood of a Gaussian distribution for data points 
    :math:`(x, y)` with a model function :math:`y=f(x, *p)` depending
    on a set of parameters :math:`*p` and a possibly parameter-dependent
    covariance matrix :math:`V(x, f(x, *p))` of the x and y data:

    .. math:: 
      -2\ln {\cal L} = \chi^2(y, V^{-1}, f(x, *p) \,) 
      + \ln(\, \det( V(x, f(x, *p) ) \,)

    In the absence of parameter-dependent components of the covariance
    matrix, the last term is omitted and the cost function is identical
    to the classical :math:`\chi^2`. 
    For the evaluation of the cost function an efficient approach based
    on the "Cholesky decomposition" of the covariance matrix in 
    a product of a triangular matrix and its transposed is used:

    .. math::
       V = L L^T.    

    The value of the cost function 

    .. math::
      \chi^2 = {r}\cdot (V^{-1}{r}) ~~with~~ r = y - f(x,*p)

    is then calculated by solving the linear equation  

   .. math::
      V X = r, ~i.e.~ X=V^{-1} r ~and~ \chi^2= r \cdot X   

   with the linear-equation solver *scipy.linalg.cho_solve(L,x)*
   for Cholesky-decomposed matrices, thus avoiding the costy 
   calculation of the inverse matrix.
    

   The determinant, if needed, is efficiently calculated by taking 
   the product of the diagonal elements of the matrix L,

    .. math::
      \det(V) = 2 \, \prod L_{i,i}
    
    Input:

    - outer: pointer to instance of calling class
    - model: model function f(x, \*par)
    - use_neg2logL: use full -2log(L) instead of chi2 if True

    __call__ method of this class is called by iminuit

    Data members:

    - ndof: degrees of freedom 
    - nconstraints: number of parameter constraints
    - gof: chi2-value (goodness of fit)
    - use_neg2logL: usage of full 2*neg Log Likelihood
    - quiet: no printout if True    

    Methods:

    - model(x, \*par)
    """
 
    def __init__(self, outer, 
                 model,
                 use_neg2logL=False):

      from iminuit.util import make_func_code

      # data object of type xyDataContainer
      self.data = outer.data
      if not isinstance(self.data, mnFit.xyDataContainer):
        raise ValueError(
          " !!! mnFit.xLSqCost: expecting data container of type 'mnFit.xyDataContainer'")
      self.model = model
      self.quiet = outer.quiet
      # use -2 * log(L) of Gaussian instead of Chi2
      #  (only different from Chi2 for parameter-dependent uncertainties)
      self.use_neg2logL = use_neg2logL
      
      # set proper signature of model function for iminuit
      self.pnams = outer.pnams
      self.func_code = make_func_code(self.pnams)
      self.npar = outer.npar
      
      # take account of constraints 
      self.constraints = outer.constraints
      self.nconstraints = len(self.constraints)
      self.ndof = len(self.data.y) - self.npar + self.nconstraints + outer.nfixed

      # flag to control final actions in cost function
      self.final_call = False


    def __call__(self, *par):  
      # called iteratively by minuit

      # cost function is extended chi2:
      #   add normalization term if uncertainties depend on model 

      nlL2 = 0. # initialize -2*ln(L)
      #  first, take into account possible parameter constraints  
      if self.nconstraints:
        for c in self.constraints:
          p_id = c[0]
          r = ( par[p_id] - c[1]) / c[2] 
          nlL2 += r*r

      # calculate residual of data wrt. model
      model_values = self.model(self.data.x, *par)
      _r = self.data.y - model_values

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
        
        # add parameter-dependent normalization term if needed and wanted
        if self.data.needs_dynamicErrors and self.use_neg2logL:
          nlL2 += np.sum(np.log(self.data.err2))

      # provide model values to data object
      self.data.model_values = model_values       

      return nlL2
    
  # --- end definition of class xLSqCost ----

  #
  # --- special code for indexed Fit (xFit)
  #      

  def set_xOptions(self,
              relative_refers_to_model=None,
              run_minos=None,
              use_negLogL=None,
              quiet=None):

    """Define options for indexed fit

       Args:
        - rel. errors refer to model else data
        - run minos else don*t run minos
        - use full neg2logL
        - don*t provide printout else verbose printout 
    """
    if relative_refers_to_model is not None:
      self.refModel = relative_refers_to_model
      self.options["refModel"][0] = int(relative_refers_to_model)
    if run_minos is not None:   
      self.run_minos = run_minos
      self.options["run_minos"][0] = int(run_minos)
    if use_negLogL is not None:   
      self.use_negLogL = use_negLogL
      self.options["use_negLogL"][0] = int(use_negLogL)
    if quiet is not None:
      self.quiet = quiet

  def init_xData(self, x,             
                 e=None, 
                 erel=None,
                 cabs=None, crel=None,
                 names=None):

    """initialize data object

    Args:
      -  x:      data values
      -  s:      independent uncertainties x
      -  srel:   independent relative uncertainties x
      -  cabs:   correlated abolute uncertainties x
      -  crel:   correlated relative uncertainties x
    """
    
    # create data object and pass all input arguments
    self.xData = self.xDataContainer(self, x, e, 
                                     erel, cabs, crel,
                                     names = names, 
                                     quiet=self.quiet)
    self.data = self.xData
    # set flags for steering of fit process in do_fit()
    self.iterateFit = self.xData.has_rel_Errors and self.refModel


  def init_xFit(self, model, model_kwargs=None, p0=None, dp0=None,
                 constraints=None,
                 fixPars=None,
                 limits=None):
    """initialize fit object

    Args:
      - model: model function f(x; \*par)
      - model_kwargs: optional, fit parameters if not from model signature 
      - p0: np-array of floats, initial parameter values 
      - dp0: array-like, initial guess of parameter uncertainties (optional)
      - constraints: (nested) list(s): [parameter name, value, uncertainty] 
        or [parameter index, value, uncertainty]
      - limits: (nested) list(s): [parameter name, min, max] 
        or [parameter index, min, max]
    """

    if self.xData is None: 
      raise ValueError(
        ' !!! mnFit.init_xFit: no data object defined - call init_data()')

    # get parameters of model function to set start values for fit
    if model_kwargs is None:
      args, model_kwargs = get_functionSignature(model)

    par = (model_kwargs, p0, dp0, constraints, fixPars, limits)
    self._setupFitParameters(*par) 

    # create cost function
    self.costf = self.indexedCost(self,
                           model,
                           use_neg2logL= self.use_negLogL)
    self._setupMinuit(model_kwargs) 

      
  class xDataContainer:
    """
    Handle data and uncertainties and  
    build covariance matrices from components

    Args:
      - outer:   pointer to instance of calling object
      - x:       abscissa of data points ("x values")
      - e:       independent uncertainties 
      - erel:   independent relative uncertainties x
      - cabs:   correlated abolute uncertainties x
      - crel:   correlated relative uncertainties x
      - quiet:   no informative printout if True

    Public methods:
      - init_dynamicErrors():
      - get_Cov(): final covariance matrix (incl. proj. x)  
      - get_iCov(): inverse covariance matrix
      - plot(): provide a figure with representation of data
 
    Data members:  
      * copy of all input arguments
      * cov: covariance matrix
      * iCov: inverse of covariance matrix
    """
    def __init__(self, outer,
                 x, e, erel, cabs, crel,
                 names = None,
                 quiet=True):
      # save poiner to calling class
      self.outer = outer

      # assume simple case w.o. cov.mat.
      self.needs_covariance = False 

      self.xNames = names # set parameter names
      
      nd = len(x)
      # store input data as numpy float arrays, ensure length nd if needed
      self.x = np.asfarray(x)         # abscissa - "x values"
      self.e = np.asfarray(e)       # independent uncertainties y
      if self.e.ndim == 0:
          self.e = self.e * np.ones(nd)
      elif self.e.ndim == 2:
          self.needs_covariance=True
      if erel is not None:
        self.erel = np.asfarray(erel) # independent relative uncertainties x
      else:
        self.erel = None
      if cabs is not None:
        self.cabs = np.asfarray(cabs) # correlated abolute uncertainties x
        if self.cabs.ndim == 0:
          self.cabs = self.cabs * np.ones(nd)
      else:
        self.cabs = None
      if crel is not None:   
        self.crel = np.asfarray(crel) # correlated relative uncertainties x
      else:
        self.crel = None
      self.quiet = quiet      # no informative printout if True

      self.nd = nd
      self.dlength = nd
      self.model = None # no model defined yet
      self.model_values = None

      # set flags for steering of fit process in do_fit()
      self.rebulildCov = None
      self.has_rel_Errors = erel is not None or crel is not None
      self.needs_covariance = self.needs_covariance or \
        self.cabs is not None or self.crel is not None


      # build (initial) covariance matrix (without x-errors)
      if self.needs_covariance:
        err2 = _build_CovMat(self.nd, self.e, self.erel,
                             self.cabs, self.crel, self.x)
      else:
        err2 = _build_Err2(self.e, self.erel, self.x)

      # initialize uncertainties and covariance matrix,
      self._initialCov(err2)
      # sets: 
      #   self.covx: covariance matrix of x
      #   self.covy: covariance matrix of y uncertainties
      #   self.cov: full covariance matrix incl. projected x
      #   self.iCov: inverse of covariance matrix
      #   self.err2: array of squared uncertainties
      #   self.iErr2: 1./self.err2
      
    def _initialCov(self, err2):
      """Build initial (static) covariance matrix for y-errors
      (for pre-fit) and calculate inverse matrix
      """
      if err2.ndim == 2:
       # got a covariance matrix, need inverse
        self.needs_covariance = True
        self.cov = err2
        self.iCov = linalg.inv(err2)
        self.err2 = np.diagonal(err2) # squared diagonal elements
      else:
      # got independent uncertainties
        self.err2 = err2
        self.err2y = err2
        self.iErr2 = 1./err2
        self.cov = np.diag(err2)
        self.iCov = np.diag(1./self.err2)
      # do not rebuild covariance matrix in cost function
      self.needs_dynamicErrors = False 

    def init_dynamicErrors(self):
      # method to switch on dynamic re-calculation of covariance matrix 
      self.ref_toModel = self.outer.refModel
      self.model = self.outer.costf.model
     
      self._staticCov = None
      self._staticErr2 = None
      self.iCov = None
      self.iErr2 = None

      # rebuild covariance matrix during fitting procedure
      self.needs_dynamicErrors = True    # flag for cost function
      self.final_call = False # flag for _rebuild_Cov: no storage of ycov 
      
      if self.needs_covariance:
        # build static (=parameter-independent) part of covariance matrix      
        if self.has_rel_Errors and self.ref_toModel:
          # some y-errors are parameter-independent
          self._staticCov = _build_CovMat(self.nd,
                       self.e, eabscor = self.cabs )
        else: 
          # all y-errors are parameter-independent
          self._staticCov = _build_CovMat(self.nd,
                              self.e, erel=self.erel,
                              eabscor = self.cabs, erelcor=self.crel,
                                            data=self.x)
        # build matrix of relative errors
        if self.ref_toModel and self.has_rel_Errors:
          self._covy0 = _build_CovMat(self.nd,
                             erel=self.erel,
                             erelcor=self.crel,
                             data=np.ones(self.nd))
        else:
          self._covy0 = None

      else: # no covariance needed, use simple math
        # build static (=parameter-independent) part of covariance matrix      
        if self.has_rel_Errors and self.ref_toModel:
          # only independent y-errors do not depend on parameters
          self._staticErr2 = _build_Err2(self.e)
        else: 
          # all y-errors are parameter-independent
          self._staticErr2 = _build_Err2( self.e, self.erel, self.x)
          

    def _rebuild_Err2(self, mpar):
      """
      (Re-)calculate uncertaingies 
      """
      if self._staticErr2 is not None:
        self.err2 = np.array(self._staticErr2, copy=True)
      else:
        self.err2 = np.zeros(self.nd)
      if self.ref_toModel and self.has_rel_Errors:
        _er = self.erel * self.model(self.x, *mpar)       
        self.err2 += _er * _er
        
    def _rebuild_Cov(self, mpar):
      """
      (Re-)Build the covariance matrix from components
      and caclulate its inverse
      """
     # start from pre-built parameter-independent part of Covariance Matrix
      self.cov = np.array(self._staticCov, copy=True)

     # add matrix of parameter-dependent y-uncertainties
      if self._covy0 is not None:
        _dat = self.model(self.x, *mpar)       
        self.cov += self._covy0 * np.outer(_dat, _dat)

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
  
    def get_iCov(self):
      """return inverse of covariance matrix, as used in cost function
      """
      if self.needs_covariance:
        return self.iCov
      else:
        return np.diag(1./self.err2)

    def plot(self, num='Data and Model',
                   figsize=(7.5, 6.5),                             
                   data_label='data',
                   plot_residual=False):
      """return figure with data and uncertainties
      """
#    # get data
      if plot_residual and self.model_values is not None:
        x = self.x - self.model_values
      else:  
        x = self.x
      e = self.get_Cov()
      if e.ndim == 2:
        e = np.sqrt(np.diagonal(e))
      else:
        e = np.sqrt(e)
   # draw data
      fig = plt.figure(num=num, figsize=figsize)
      idx = np.linspace(0.5, len(x)-0.5, len(x))
      plt.plot(idx, x, marker='x', linestyle='', color='grey', alpha=0.5)
      plt.errorbar(idx, x, e, fmt=".", capsize=3.0, label=data_label)
      plt.xlim(0., len(idx))
      if self.xNames is not None:
         plt.xticks(ticks=idx, labels=self.xNames)
      else: 
         plt.xticks(ticks=idx, labels=range(len(idx)))

      return fig
      
  # define custom cost function for iminuit
  class indexedCost:
    """
    Custom e_x_tended Least-SQuares cost function with 
    dynamically updated covariance matrix and -2log(L) 
    correction term for parameter-dependent uncertainties.

    The default cost function is twice the negative logarithm 
    of the likelihood of a Gaussian distribution for data points 
    :math:`(x, y)` with a model function :math:`y=f(x, *p)` depending
    on a set of parameters :math:`*p` and a possibly parameter-dependent
    covariance matrix :math:`V(x, f(x, *p))` of the x and y data:

    .. math:: 
      -2\ln {\cal L} = \chi^2(x, V^{-1}, x(*p) \,) 
      + \ln(\, \det( V(x, x(*p) ) \,)

    In the absence of parameter-dependent components of the covariance
    matrix, the last term is omitted and the cost function is identical
    to the classical :math:`\chi^2`. 
    For the evaluation of the cost function an efficient approach based
    on the "Cholesky decomposition" of the covariance matrix in 
    a product of a triangular matrix and its transposed is used:

    .. math::
       V = L L^T.    

    The value of the cost function 

    .. math::
      \chi^2 = {r}\cdot (V^{-1}{r}) ~~with~~ r = x - x(*p)

    is then calculated by solving the linear equation  

   .. math::
      V X = r, ~i.e.~ X=V^{-1} r ~and~ \chi^2= r \cdot X   

   with the linear-equation solver *scipy.linalg.cho_solve(L,x)*
   for Cholesky-decomposed matrices, thus avoiding the costy 
   calculation of the inverse matrix.
    

   The determinant, if needed, is efficiently calculated by taking 
   the product of the diagonal elements of the matrix L,

    .. math::
      \det(V) = 2 \, \prod L_{i,i}
    
    Input:

    - outer: pointer to instance of calling class
    - model: model function calulating the data x(\*par)
    - use_neg2logL: use full -2log(L) instead of chi2 if True

    __call__ method of this class is called by iminuit

    Data members:

    - ndof: degrees of freedom 
    - nconstraints: number of parameter constraints
    - gof: chi2-value (goodness of fit)
    - use_neg2logL: usage of full 2*neg Log Likelihood
    - quiet: no printout if True    

    Methods:

    - model(x, \*par)
    """
 
    def __init__(self, outer, 
                 model,
                 use_neg2logL=False):

      from iminuit.util import make_func_code

      # data object of type xyDataContainer
      self.data = outer.data
      if not isinstance(self.data, mnFit.xDataContainer):
        raise ValueError(
          " !!! mnFit.indexedCost: expecting data container of type 'mnFit.xDataContainer'")
      self.model = model
      self.quiet = outer.quiet
      # use -2 * log(L) of Gaussian instead of Chi2
      #  (only different from Chi2 for parameter-dependent uncertainties)
      self.use_neg2logL = use_neg2logL
      
      # set proper signature of model function for iminuit
      self.pnams = outer.pnams
      self.func_code = make_func_code(self.pnams)
      self.npar = outer.npar

      # take account of constraints 
      self.constraints = outer.constraints
      self.nconstraints = len(self.constraints)
      self.ndof = len(self.data.x) - self.npar + self.nconstraints + outer.nfixed

      # flag to control final actions in cost function
      self.final_call = False


    def __call__(self, *par):  
      # called iteratively by minuit

      # cost function is extended chi2:
      #   add normalization term if uncertainties depend on model 

      nlL2 = 0. # initialize -2*ln(L)
      #  first, take into account possible parameter constraints  
      if self.nconstraints:
        for c in self.constraints:
          p_id = c[0]
          r = ( par[p_id] - c[1]) / c[2] 
          nlL2 += r*r

      # calculate residual of data wrt. model
      model_values = self.model(self.data.x, *par)
      _r = self.data.x - model_values

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
        
        # add parameter-dependent normalization term if needed and wanted
        if self.data.needs_dynamicErrors and self.use_neg2logL:
          nlL2 += np.sum(np.log(self.data.err2))

      # provide model values to data object
      self.data.model_values = model_values       

      return nlL2


  #
  # --- special code for histogram Fit
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
      self.options["run_minos"][0] = int(run_minos)
    if use_GaussApprox is not None:   
      self.use_GaussApprox = use_GaussApprox
      self.options["use_GaussApprox"][0] = int(use_GaussApprox)
    if fit_density is not None:
      self.fit_density = fit_density
      self.options["fit_density"][0] = int(fit_density)
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
    self.hData = self.histDataContainer(self,
                               bin_contents, bin_edges,
                               DeltaMu,
                               quiet=self.quiet)
    self.data = self.hData
    
  def init_hFit(self, model, model_kwargs=None, p0=None, dp0=None,
                constraints=None,
                fixPars = None, 
                limits=None):
    """initialize fit object

    Args:
      - model: model density function f(x; \*par)
      - model_kwargs: optional, fit parameters if not from model signature 
      - p0: np-array of floats, initial parameter values 
      - dp0: array-like, initial guess of parameter uncertainties (optional)
      - constraints: (nested) list(s): [parameter name, value, uncertainty] 
        or [parameter index, value, uncertainty]
      - fix parameter(s) in fit: list of parameter names or indices
      - limits: (nested) list(s): [parameter name, min, max] 
        or [parameter index, min, max]
    """

    if self.hData is None: 
      raise ValueError(
        ' !!! mnFit.init_hFit: no data object defined - call init_data()')
    
    # get parameters of model function to set start values for fit
    if model_kwargs is None:
      args, model_kwargs = get_functionSignature(model)

    par = (model_kwargs, p0, dp0, constraints, fixPars, limits)
    self._setupFitParameters(*par)

    # create cost function
    self.costf = self.hCost(self, model,
                            use_GaussApprox=self.use_GaussApprox,
                            density = self.fit_density)
    self._setupMinuit(model_kwargs) 
      
  class histDataContainer:
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

      Methods:

      - plot(): return figure with histogram of data and uncertainties
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

      self.outer = outer
      self.contents = bin_contents
      self.nbins=len(bin_contents)
      self.dlength = self.nbins
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
      
    def plot(self, num='histData and Model',
                   figsize=(7.5, 6.5),                             
                   data_label='Binned data',
                   plot_residual=False):

      """return figure with histogram data and uncertainties
      """

      w = self.edges[1:] - self.edges[:-1]
      fig = plt.figure(num=num, figsize=figsize)
      if self.model_values is not None:
        if plot_residual:
          mconts = np.zeros(self.nbins)
          bconts = self.contents - self.model_values
        else:
          mconts = self.model_values
          bconts = self.contents
      plt.bar(self.centers, bconts,
              align='center', width = w,
              facecolor='cadetblue', edgecolor='darkblue', alpha=0.66,
              label = data_label)

      # set and plot error bars
      if self.model_values is not None:
      # best-fit model values avaiable
        if not self.outer.use_GaussApprox:
          # show Poisson Confidence Intervals
          ep = []
          em = []
          for i in range(self.nbins):
            l = (self.model_values[i] + np.abs(self.DeltaMu[i]))      
            m, p = self.Poisson_CI(l, sigma=1.)
            ep.append(p-l)
            em.append(l-m)
          plt.errorbar(self.centers, mconts,
                       yerr=(em, ep), fmt=' ', 
                       ecolor='olive', capsize=3,  
                     alpha=0.8)
        else: # show symmetric error bars 
          ep = np.sqrt(self.model_values + np.abs(self.DeltaMu))
          em = [ep[i] if self.model_values[i]-ep[i]>0. else self.model_values[i]
                for i in range(len(ep))]
          plt.errorbar(self.centers, self.model_values,
                       yerr=(em, ep), fmt=' ', 
                       ecolor='olive', elinewidth=2, alpha=0.8)
      else: # no model values available (yet), show error bars related to data      
        ep = np.sqrt(self.contents + np.abs(self.DeltaMu))
        em = [ep[i] if self.contents[i]-ep[i]>0. else self.contents[i] for i in range(len(ep))]      
        plt.errorbar(self.centers, bconts,
                   yerr=(em, ep),
                   fmt='_', color='darkblue', markersize=15,
                   ecolor='darkblue', alpha=0.8)
      return fig

    @staticmethod
    def Poisson_CI(lam, sigma=1.):
      """
      determine one-sigma Confidence Interval around the 
      mean lambda of a Poisson distribution, Poiss(x, lambda). 

      The method is based on delta-log-Likelihood (dlL) 
      of the Poission likelihood 

      Args:
       - lam: mean of Poission distribution
       - cl: desired confidence level
       - sigma: alternatively specify an n-sigma interval
      """

      # functions
      def nlLPoisson(x, lam):
        """negative log-likelihood of Poissoin distrbution
        """
        return lam - x*np.log(lam) + loggamma(x+1.)

      def f(x, lam, dlL):
        """Delta log-L - offset, input to Newton method
        """
        return nlLPoisson(x, lam) - nlLPoisson(lam, lam) - dlL

      dlL = 0.5*sigma*sigma 

      # for dlL=0.5, there is only one intersection with zero for lam<1.8
      dl = 1.2*np.sqrt(lam)
      dlm = min(dl, lam)
      cp = newton(f, x0=lam+dl, x1=lam, args=(lam, dlL))
      try: # may not converge for small lambda, as there is no intersection < lam
        cm = newton(f, x0=lam-dlm, x1=lam, args=(lam, dlL))
      except:
        cm = 0.
      if (cp-cm)<lam/100.: # found same intersection,
        cm = 0.            #  set 1st one to 0.  
 
      return cm, cp            
    
  # --- cost function for histogram data
  class hCost:
    """    Cost function for binned data

    The __call__ method of this class is called by iminuit.

    The default cost function to minimoze is twice the negative 
    log-likelihood of the Poisson distribution generalized to 
    continuous observations x by replacing k! by the gamma function:

    .. math::
        cost(x;\lambda) = 2 \lambda (\lambda - x*\ln(\lambda) + \ln\Gamma(x+1.))

    Alternatively, the Gaussian approximation is available:

    .. math::
        cost(x;\lambda) = (x - \lambda)^2 / \lambda + \ln(\lambda)
           
    The implementation also permits to shift the observation x by an
    offset to take into account corrections to the number of observed
    bin entries (e.g. due to background or efficiency corrections):
    x -> x-deltaMu with deltaMu = mu - lambda, where mu is the mean
    of the shifted Poisson or Gauß distribution.  

    The number of bin entries predicted by the model density is calculated
    by an approximate integral over the respective bin ranges using the
    Simpson rule. 

    To judge the level of agreement of model density and histogram data, 
    a "goodness-of-fit" (*gof*) value is calulated as the likelihood-ratio of
    the model w.r.t. the data and the so-called "saturated model" describing 
    the data perfectly, i.e.  :math:`cost_{sat}(x) = cost(x; \lambda=x)`.
    If the bin entries are sufficiently large, *gof* converges to the standard 
    *chi2* value. 

    Input:

    - outer: pointer to instance of calling class
    - model: model function f(x, \*par)
    - use_GaussApprox, bool: use Gaussian approximation 
    - density, bool: fit a normalised density; if False, an overall
      normalisation must be provided in the model function

    Data members:

    - ndof: degrees of freedom 
    - nconstraints: number of parameter constraints
    - gof: goodness-of-fit as likelihood ratio w.r.t. the 'saturated model'

    External references:

    - model(x, \*par): the model function 
    - data: pointer to instance of class histData
    - data.model_values: bin entries calculated by the best-fit model
    """

    def __init__(self, outer, 
                 model,
                 use_GaussApprox=False, density= True):
      from iminuit.util import make_func_code

      # data object of type histDataContainter
      self.data = outer.hData
      if not isinstance(self.data, mnFit.histDataContainer):
        raise ValueError(
          " !!! mnFit.hCost: expecting data container of type 'histDataContainer'")
      
      self.model = model
      self.density = density
      self.GaussApprox = use_GaussApprox
      self.quiet = outer.quiet
      
      # set proper signature of model function for iminuit
      self.pnams = outer.pnams
      self.func_code = make_func_code(self.pnams)
      self.npar = outer.npar
 
      self.constraints = outer.constraints
      self.nconstraints =len(self.constraints)
      # take account of constraints in degrees of freedom 
      self.ndof = self.data.nbins-self.npar+self.nconstraints+outer.nfixed

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
        
    def __call__(self, *par):
      # called iteratively by minuit

      # cost function is likelihood of shifted Poisson or Gauss approximation

      # - first, take into account possible parameter constraints  
      n2lL= 0.
      if self.nconstraints:
        for c in self.constraints:
          p_id = c[0]
          r = ( par[p_id] - c[1]) / c[2] 
          n2lL += r*r

      # - calculate 2*negLogL Poisson;
      #  model prediction as approximate integral over bin
      model_values = self.norm * self.integral_overBins(
        self.data.lefts, self.data.rights,
        self.model, *par) 
      # 

      n2lL += np.sum(
        self.n2lLcost( self.data.contents - self.data.DeltaMu, 
                      model_values + np.abs(self.data.DeltaMu) ) )
       
      if self.final_call:
        if self.GaussApprox:
          # return standard chi^2
          self.gof = n2lL-np.log(model_values+np.abs(self.data.DeltaMu)).sum() 
        else: 
          # store goodness-of-fit (difference of nlL2 w.r.t. saturated model)
          n2lL_saturated = np.sum(
            self.n2lLcost(
              self.data.contents - self.data.DeltaMu, 
              self.data.contents + np.abs(self.data.DeltaMu) + 0.005) )
        #                                !!! const. 0.005 to avoid log(0.) 
          self.gof =  n2lL - n2lL_saturated

        # provide model values and model-related uncertainties to data object
        self.data.model_values = model_values       
        
       # return 2 * neg. logL
      return n2lL
    
    @staticmethod
    def n2lLPoisson(x, lam):  
      """
      neg. logarithm of Poisson distribution for real-valued x

      """
      return 2.*(lam - x*np.log(lam) + loggamma(x+1.))

    @staticmethod
    def n2lLsPoisson(xk, lam, mu):  
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
  # --- special code for fit with user-supplied cost function
  #     or pdf for neg. log-likelihood fit 
  #      

  def init_mlData(self, x):
    """
    initialize data object

    Args:
    - x, array of floats
    """
    
    # create data object and pass all input arguments
    self.mlData = self.mlDataContainer(self, x)
    self.data = self.mlData


  def init_mnFit(self, userFunction, model_kwargs=None, p0=None, dp0=None,
                       constraints=None, fixPars=None, limits=None):
    """initialize fit object for simple minuit fit with
    * with user-supplied cost function or
    * a probability density function for an unbinned neg. log-L fit

    Args:
      - costFunction: cost function or pdf 
      - p0: np-array of floats, initial parameter values 
      - model_kwargs: optional, fit parameters if not from model signature 
      - dp0: array-like, initial guess of parameter uncertainties (optional)
      - parameter constraints: (nested) list(s): [parameter name, value, uncertainty] 
      - fix parameter(s) in fit: list of parameter names or indices
      - limits: (nested) list(s): [parameter name, min, max] or
        [parameter index, min, max]
    """
    if self.data is None and self.fit_type != "user":
      raise ValueError(
        ' !!! mnFit.init_mnFit: no data object defined - call init_data()')

    # get parameters of model function to set start values for fit
    if model_kwargs is None:
      args, model_kwargs = get_functionSignature(userFunction)

    par = (model_kwargs, p0, dp0, constraints, fixPars, limits)
    self._setupFitParameters(*par) 

    #set up cost function for iminuit
    self.costf = self.mnCost(self, userFunction)
    self._setupMinuit(model_kwargs) 

  def set_mnOptions(self, run_minos=None, neg2logL=None, quiet=None):
    """Define options for minuit fit with user cost function

    Args:

    - run_minos: run minos profile likelihood scan
    - neg2logL: cost function is -2 negLogL
    """
    if run_minos is not None:   
      self.run_minos = run_minos
      self.options["run_minos"][0] = int(run_minos)
    if neg2logL is not None:
      self.neg2logL = neg2logL
      self.options["neg2logL"][0] = int(neg2logL)
      if self.neg2logL:
        self.ErrDef = 1.
      else:
        self.ErrDef = 0.5
    if quiet is not None:
      self.quiet = quiet

  class mlDataContainer:
    """
      Container for general (indexed) data

      Data Members:

      - x, array of floats: data


      Methods:

      -plot(): return figure with representation of data
    """
    
    def __init__(self, outer, x):
      """ 
      store data

      Args:
      - x, array of floats
      - quiet: boolean, controls printed output

      """

      self.x = np.asarray(x)
      self.dlength = len(self.x)

      
    def plot(self, num='indexed data',
                   figsize=(7.5, 6.5),                             
                   data_label='Data',
                   plot_residual=False):
      """return figure with histogram data and uncertainties
      """

      fig = plt.figure(num=num, figsize=figsize)

      if plot_residual:
        print(' !!! mnFit.mlData.plot: plotting residuals not possible for user ML fit')
        
      if self.x is None:
        print(' !!! mnFit.mlData.plot: no data object defined')
        return fig
      
      mn = min(self.x)
      mx = max(self.x)
      ymn = 0.
      ymx = 0.25/(mx-mn)
      plt.vlines(self.x, ymn, ymx, lw=1, color='grey', alpha=0.5, label=data_label)
      
      return fig

      
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
                 userFunction):
      from iminuit.util import make_func_code

      self.quiet = outer.quiet
      self.data = outer.data
      if self.data is None:
        self.cost = userFunction
        if not self.quiet:
          print("*==* mnFit.mnCost: fit with user-supplied cost function'")          
      else:
        if not isinstance(self.data, mnFit.mlDataContainer):
          raise ValueError(
            " !!! mnFit.mnCost: expecting data container of type 'mlDataContainer'")
        self.model = userFunction
        self.cost = self.nlLcost
        if not self.quiet:
          print("*==* mnFit.mnCost: negLogL fit with user-defined density function'")          

      # take account of possible parameter constraints
      self.constraints = outer.constraints
      self.nconstraints = len(self.constraints)
      self.ErrDef = outer.ErrDef
      self.sf_lL = -2.*self.ErrDef
      
      # set proper signature of model function for iminuit
      self.pnams = outer.pnams
      self.func_code = make_func_code(self.pnams)
      self.npar = outer.npar

      # for this kind of fit, some input and ouput quantities are not known
      self.gof = None
      self.ndof = None
      
    def __call__(self, *par):
      cost = 0.
      # add constraints to cost
      if self.nconstraints:
        for c in self.constraints:
          p_id = c[0]
          r = ( par[p_id] - c[1]) / c[2] 
          cost += r*r
        cost *= self.ErrDef 
      # called iteratively by minuit
      return cost + self.cost(*par)

    def nlLcost(self, *par):
      """negative log likelihood of data and user-defined PDF and 
      """
       # check if PDF is nomalized to 1
       # ...

      lL= np.sum( np.log( self.model(self.data.x, *par) ) )
      return self.sf_lL * lL
        
 # --- end definition of class mnCost ----

 #
 # --- common code for all fit types
 #

  def _setupFitParameters(self, model_kwargs, p0, dp0, constraints, fixPars, limits):
    """set up parameters needed for Minuit and cost function
    """

    # get parameter names from kwargs of model function
    self.pnams = list.copy(list(model_kwargs.keys()))
    self.npar = len(self.pnams)
    # dictionary assigning parameter name to index
    self.pnam2id = {
      self.pnams[i] : i for i in range(0,self.npar) } 

    # process and store parameter constraints (used in cost function)
    self.setConstraints = constraints    
    self.constraints = []
    if constraints is not None:
      if len(np.shape(constraints))==2:
        for c in constraints:
          # name to parameter id
          if type(c[0])==type(' '):
            c[0] = self.pnam2id[c[0]]
          self.constraints.append(c)
      else:
        # name to parameter id
        if type(constraints[0])==type(' '):
          constraints[0] = self.pnam2id[constraints[0]]
        self.constraints.append(constraints)
    self.nconstraints = len(self.constraints)

    # set initial parameters for fit (used by minuit) 
    if p0 is not None:
      for i, pnam in enumerate(self.pnams):
        model_kwargs[pnam] = p0[i]    

    # set initial uncertainties (optional)
    self.dp0 = dp0
    
    # store informations on parameter limits (used by minuit)
    self.setLimits = limits    
    self.limits = []
    if limits is not None:
      self.limits=[ [None, None]] * len(self.pnams)
      if len(np.shape(limits))==2:
        for l in limits:
          if type(l[0])==type(' '):
            p_id = self.pnam2id[l[0]]
          else:
            p_id = l[0]
          self.limits[p_id] = [l[1], l[2]]          
      else:
        if type(limits[0])==type(' '):
          p_id = self.pnam2id[limits[0]]
        else:
          p_id = limits[0]
        self.limits[p_id]=[limits[1], limits[2]]          

    # store information on fixed parameters 
    self.fixPars = fixPars    
    self.fixedPars = [ False ] * self.npar
    if fixPars is not None:
      # get parameter names or indices to fix
      if len(np.shape(fixPars))==1:
        for f in fixPars:
          if type(f)==type(' '):
            p_id = self.pnam2id[f]
          else:
            p_id = f
          self.fixedPars[p_id] = True          
        self.nfixed = len(fixPars)
      else:
        if type(fixPars)==type(' '):
          p_id = self.pnam2id[fixPars]
        else:
          p_id = fixPars
        self.fixedPars[p_id]=True          
        self.nfixed = 1
    # get parameter names or indices of fixed parameters
      self.freeParNams = []
      self.fixedParNams = []
      for i, fixed in enumerate(self.fixedPars):
        if not fixed: self.freeParNams.append(self.pnams[i])
        if fixed: self.fixedParNams.append(self.pnams[i])
    else:
      self.freeParNams = self.pnams  
      self.fixedParNams = []  

  def _setupMinuit(self, model_kwargs):
    """create Minuit object (depends on Minuit version)
    """
    if self.iminuit_version < '2':
      if self.quiet:
        print_level=0
      else:
        print_level=1
      if self.dp0 is not None:
        for i, pnam in enumerate(self.pnams):
          model_kwargs['error_' + pnam] = self.dp0[i]
      if self.setLimits is not None:
        for i, pnam in enumerate(self.pnams):
          model_kwargs['limit_' + pnam] = self.limits[i]
      if self.fixPars is not None:
        for i, pnam in enumerate(self.pnams):
          model_kwargs['fix_' + pnam] = self.fixedPars[i]          
      self.minuit = Minuit(self.costf,
                           errordef=self.ErrDef,
                           print_level=print_level,
                           **model_kwargs )
    else:
      self.minuit = Minuit(self.costf,
                           *model_kwargs.values(),
                           name=self.pnams )
      self.minuit.errordef = self.ErrDef
      if self.quiet:
        self.minuit.print_level = 0
      if self.dp0 is not None:
        self.minuit.errors = self.dp0
      if self.setLimits is not None:
        self.minuit.limits = self.limits
      if self.fixPars is not None:
        for i, pnam in enumerate(self.pnams):
          if self.fixedPars[i]: self.minuit.fixed[pnam]=True 

  def _storeResult(self):
  # collect results as numpy arrays
    # !!! this part depends on iminuit version !!!    
    m=self.minuit
    minCost = m.fval                        # minimum value of cost function
    nfpar = m.nfit                          # numer of free parameters
    ndof = self.costf.ndof                  # degrees of freedom
    if self.iminuit_version < '2':
      parnames = m.values.keys()            # parameter names
      parvals = np.asarray(m.values.values()) # best-fit values
      parerrs = np.asarray(m.errors.values()) # parameter uncertainties
      cov = np.array(m.matrix())            # cov. matrix of free parameters
    else:
    # vers. >=2.0 
      parnames = m.parameters       # parameter names
      parvals = np.asarray(m.values)  # best-fit values
      parerrs = np.asarray(m.errors)  # parameter uncertainties
      cov = np.asarray(m.covariance)  # covariance matrix of all(!) parameters
      # produce reduced covariance matrix for free parameters only
      if self.nfixed != 0:
        for i in range(len(parnames)-1, -1, -1): # start from largest index and work back
          if self.fixedPars[i]:
            cov = np.delete(np.delete(cov, i, 0 ), i, 1)
    npar=len(parnames)             # number of parameters

    if self.minosResult is not None and self.minos_ok:
      pmerrs = [] 
    #  print("MINOS errors:")
      if self.iminuit_version < '2':
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
    # * flag for fixed parameters in self.fixedPars
    # * names of free parameters in self.freeParNams
    #   chi2 at best-fit point (possibly different from minCost)
    self.GoF = self.costf.gof  
    #   parameter values at best-fit point
    self.ParameterValues = np.array(parvals, copy=True)
    self.freeParVals = np.array(
      [ parvals[i] for i in range(npar) if not self.fixedPars[i] ])
    self.freeParErrs = np.array(
      [ parerrs[i] for i in range(npar) if not self.fixedPars[i] ])
    # fixed parameter names and values
    self.fixedParVals = np.array(
      [ parvals[i] for i in range(npar) if self.fixedPars[i] ])
    # * names of fixed parameters in self.fixedParNams
    #   number of degrees of freedom
    self.NDoF = ndof  
    #   symmetric uncertainties
    self.MigradErrors = np.array(parerrs, copy=True)
    #   covariance and correlation matrices
    self.CovarianceMatrix = np.array(cov, copy=True)
    perrs = np.sqrt(np.diagonal(cov))
    self.CorrelationMatrix = cov/np.outer(perrs, perrs)
    #   1-sigma (68% CL) range in self.OneSigInterval

    # build a convenient result dictionary
    if self.nfixed ==0:
      rtuple = (self.ParameterValues,
                self.OneSigInterval,
                self.CorrelationMatrix,
                self.GoF,
                self.ParameterNames )
    else:
       rtuple= (np.concatenate( (self.freeParVals, self.fixedParVals) ),
                self.OneSigInterval,
                self.CorrelationMatrix,
                self.GoF,
                np.concatenate( (self.freeParNams, self.fixedParNams) ) )
    keys = ( 'parameter values',
             'confidence intervals',
             'correlation matrix',
             'goodness-of-fit',
             'parameter names' )
    # build dictionary
    self.ResultDictionary = {k:rtuple[i] for (i, k) in enumerate(keys)}
    
  def getResult(self):
    """return result dictionary
    """
    if self.ResultDictionary is not None:
      return self.ResultDictionary
    else:
      raise RuntimeError(
        " !!! mnFit.getResult: no results available - run fit first")
       
  @staticmethod
  def getFunctionError(x, model, pvals, covp, fixedPars):
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
    # number of variable parameters 
    nvpar = len(dp)
    #   derivative df/dp_j at each x_i
    dfdp = np.empty( (nvpar, len(x)) )
    p_plus = np.array(pvals, copy=True)
    p_minus = np.array(pvals, copy=True)
    k=0
    for j in range(len(pvals)):
      if not fixedPars[j]:
        p_plus[j] = pvals[j] + dp[k]
        p_minus[j] = pvals[j] - dp[k]
        dfdp[k] = 0.5 / dp[k] * (
                    model(x, *p_plus) - 
                    model(x, *p_minus) )
        p_plus[j] = pvals[j]
        p_minus[j] = pvals[j]
        k+=1
    #   square of uncertainties on function values
    Delta= np.empty(len(x))
    for i in range(len(x)):
      Delta[i] = np.sum(np.outer(dfdp[:,i], dfdp[:,i]) * covp)
    return np.sqrt(Delta) 
  

  def do_fit(self):
    """perform all necessary steps of fit sequence
    """
    if self.data is None and self.fit_type != "user":
      raise ValueError(
        ' !!! mnFit: no data object defined - call init_data()')
    if self.costf is None:
      raise RuntimeError(
        '!!! mnFit: no fit object defined - call init_fit()')
    
    # summarize options
    if not self.quiet:
      if self.iterateFit:
        print( '*==* mnFit starting pre-fit')
      else: 
        print( '*==* mnFit starting fit')
      print( '  Options:')
      for key in self.options.keys():
        relevant = self.options[key][1] == "all" or \
          self.fit_type in self.options[key][1]
        if relevant:
          iopt = self.options[key][0] + 2
          print(5*" " + "- ",
              self.options[key][iopt])
      print('\n')  

    # perform (initial) fit
    try:
      self.migradResult = self.minuit.migrad()  # find minimum of cost function
      self.migrad_ok = True
    except Exception as e:
      self.migrad_ok = False
      print(e)
      raise RuntimeError(
        '*==* !!! fit with migrad failed')

    # possibly, need to iterate
    if self.iterateFit:
      if not self.quiet:
        print( '*==* mnFit iterating',
               'to account for parameter-dependent uncertainties')
      # enable dynamic calculation of covariance matrix
      self.data.init_dynamicErrors()

      # fit with dynamic recalculation of covariance matrix
      try:
        self.migradResult = self.minuit.migrad()
        self.migrad_ok = True
      except Exception as e:
        self.migrad_ok = False
        print(e)
        raise RuntimeError(
          '*==* !!! iteration of fit with migrad failed')

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


  def setPlotOptions(self):
    """Set options for nicer plotting
    """

    # to be implemented
    pass

  def plotModel(self,
                axis_labels=['x', 'y = f(x, *par)'], 
                data_legend = 'data',    
                model_legend = 'fit',
                plot_band=True,
                same_plot= False,
                plot_residual=False): 
    """
    Plot model function and data 
    
    Uses iminuitObject, cost Function (and data object)

    Args: 
      * list of str: axis labels
      * str: legend for data
      * str: legend for model 
      * plot_band: plot model confidence band if True
      * same_plot: accumulate fit results from previous call if True
      * plot_residual: plot residual w.r.t. model if True

    Returns:
      * matplotlib figure
    """
    
  # access objects
    m = self.minuit  # minuit object
    cf = self.costf  # cost function object
    d = cf.data
    npts = max(190, 5*d.dlength) # number of points for model function
    
  # retrieve fit results
    pnams = self.ParameterNames
    pvals = self.ParameterValues
    pmerrs = self.OneSigInterval
    #  symmetric errors
    perrs = (pmerrs[:,1]-pmerrs[:,0])/2.
    cor = self.CorrelationMatrix
    #  covariance matrix from correlations
    pcov = cor * np.outer(perrs, perrs)
    gof = self.GoF
    ndof = cf.ndof
    #  chi2prb
    if gof is not None:
      chi2prb = self.chi2prb(gof, ndof) 
    else:
      chi2prb = None      
    # values of free and fixed parameters
    fixedPars = self.fixedPars
    free_pvals = self.freeParVals
    free_pnams = self.freeParNams
    fixed_pnams= self.fixedParNams
    fixed_pvals= self.fixedParVals
    nfixed = len(fixed_pvals)
    
  # plot data
    fig_model = d.plot(figsize=(7.5, 6.5),
                       data_label=data_legend,
                       plot_residual=plot_residual)

  # overlay model function
    # histogram fit provides normalised distribution,
    #    determine bin widths and scale factor
    xmin, xmax = plt.xlim()
    xplt = np.linspace(xmin, xmax, npts)
    xvals = xplt 
    if self.fit_type=='hist':
      # detemine local bin width
      bwidths = np.zeros(len(xplt))
      i=0
      for j, x in enumerate(xplt):
        if x >= cf.data.rights[min(i, cf.data.nbins-1)]:
          i += 1
        bwidths[j] = cf.data.widths[min(i, cf.data.nbins-1)]
      sfac = cf.norm * bwidths
    elif self.fit_type=="xy" or self.fit_type=='ml':
      sfac = 1.
    elif self.fit_type=="indexed":
      sfac = 1.
      xvals = cf.data.x
      xplt = np.linspace(0.5, len(xvals)-0.5, len(xvals))
    else:
      print("!**! mnFit.plotModel: unknown fit type: ", self.fit_type)
      sfac = 1.
      return
    
    # do we want model or residuals ?  
    if plot_residual:
      yplt = np.zeros(len(xplt))
    else:
      yplt = cf.model(xvals, *pvals)

    #  plot model line
    if self.fit_type == "indexed":
      plt.errorbar(xplt, yplt, fmt=' ', xerr=0.4, label=model_legend,
                     alpha=0.7, elinewidth=2.5, color='darkorange')
    else:
      plt.plot(xplt, yplt*sfac, label=model_legend,
             linestyle='dashed', linewidth=2.5, alpha = 0.7)
   ##          color='darkorange')
    plt.xlabel(axis_labels[0], size='x-large')
    plt.ylabel(axis_labels[1], size='x-large')
    plt.grid()

    # draw error band around model function
    if plot_band:
      if self.fit_type != "indexed":
        DeltaF = self.getFunctionError(xplt, cf.model, pvals, pcov, fixedPars)
        plt.fill_between(xplt, sfac*(yplt+DeltaF),
                             sfac*(yplt-DeltaF),
                             alpha=0.3, color='darkkhaki', label='  $\pm 1 \sigma$')
        plt.plot(xplt, sfac*(yplt+DeltaF), linewidth=1, 
                             alpha=0.4, color='darkgreen')
        plt.plot(xplt, sfac*(yplt-DeltaF), linewidth=1,
                             alpha=0.4, color='darkgreen')
      else:
        Delta = self.getFunctionError(xvals, cf.model, pvals, pcov, fixedPars)
        plt.errorbar(xplt, yplt+Delta, fmt=' ', xerr=0.4,
                     alpha=0.7, elinewidth=1.0, color='darkgreen')
        plt.errorbar(xplt, yplt-Delta, fmt=' ', xerr=0.4,
                     alpha=0.7, elinewidth=1., color='darkgreen')
        for i in range(len(xplt)):
           lbl = '$\pm 1 \sigma$' if i==0 else ''
           plt.axhspan(yplt[i]-Delta[i], yplt[i]+Delta[i],
                       xmin=(xplt[i]-0.4)/len(xplt),
                       xmax=(xplt[i]+0.4)/len(xplt),
                       alpha=0.3, color='darkkhaki', label=lbl) 
       
  # display legend with fit info 
    global fit_info
    if same_plot:
      try: fit_info
      except NameError: fit_info = []
    else:
      fit_info = []      

    #  1. parameter values and uncertainties
    pe = 2   # number of significant digits of uncertainty
    if self.minosResult is not None and self.minos_ok:
      for pn, v, e in zip(free_pnams, free_pvals, pmerrs):
        nd, _v, _e = round_to_error(v,min(abs(e[0]),abs(e[1])),nsd_e=pe)
        txt="{} = ${:#.{pv}g}^{{+{:#.{pe}g}}}_{{{:#.{pe}g}}}$"
        fit_info.append(txt.format(pn, _v, e[1], e[0], pv=nd, pe=pe))
    else:
      for pn, v, e in zip(free_pnams, free_pvals, pmerrs):
        nd, _v, _e = round_to_error(v, e[1], nsd_e=pe)
        txt="{} = ${:#.{pv}g}\pm{:#.{pe}g}$"
        fit_info.append(txt.format(pn, _v, _e, pv=nd, pe=pe))
    if nfixed:
      for pn, v in zip(fixed_pnams, fixed_pvals):
        txt="{} = {:g}  (fixed)"
        fit_info.append(txt.format(pn, v))
        
    #  2. goodness-of-fit
    if self.fit_type in ['xy', 'indexed']:
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

    # save color map and set new one - affects iminuit.draw_mncontour()
    orig_cm = plt.get_cmap()
    plt.set_cmap('tab20b')

    m = self.minuit
    fpnams = self.freeParNams
    fpvals = self.freeParVals
    fperrs = self.freeParErrs
    npar = len(fpnams)

    fsize = 3.5 if npar<=3 else 2.5
    cor_fig, axarr = plt.subplots(npar, npar,
                                  num=figname,
                                  figsize=(fsize*npar, fsize*npar),
                                  constrained_layout=True)
    
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
            plt.sca(axarr[ip, ip]) if npar!=1 else plt.sca(axarr)
            m.draw_mnprofile(fpnams[i], subtract_min=True)
            if (self.ErrDef == 1):
              ylabel='$\Delta (-2\ln\cal{L})$'
            else:
              ylabel='$\Delta (-\ln\cal{L})$'
            plt.ylabel(ylabel)
            xmn, xmx = plt.gca().get_xlim()
            # show horizontal line at self.ErrDef 
            plt.hlines(self.ErrDef, xmn, xmx, color='orange', linestyle='--')
            plt.errorbar(fpvals[i], 0., xerr=fperrs[i],
                         fmt='x', color='blue', linestyle='-', capsize=3)
          else:
            plt.sca(axarr[jp, ip])
            if self.iminuit_version <'2':
              m.draw_mncontour(fpnams[i], fpnams[j])
            else:
              m.draw_mncontour(fpnams[i], fpnams[j],
                cl=(self.Chi22CL(1.), self.Chi22CL(4.)) )
            # plot best-fit values and migrad errors
            plt.errorbar(fpvals[i], fpvals[j],
                         xerr=fperrs[i], yerr=fperrs[j],
                         fmt='x', color='darkblue',
                         ecolor='blue', capsize=3, 
                         alpha=0.66)
      # restore color map
      plt.set_cmap(orig_cm)        

      return cor_fig

    except Exception as e:
      print('*==* !!! profile and contour scan failed')
      print(e)
      return None

  def getProfile(self, pnam, range=3., npoints=30):
    """return profile likelihood of parameter pnam

    Args:
      - parameter name
      - scan range in sigma, are tuple with lower and upper value
      - number of points 
    """
    if self.iminuit_version <'2':
      print("!!! getProfile not implemented vor iminuit vers.<2")
      return
    else:
      return self.minuit.mnprofile(pnam, bound = range,
                                 size=npoints, subtract_min=True)

  def getContour(self, pnam1, pnam2, cl=None, npoints=100):
    """return profile likelihood contour of parameters pnam1 and pnam2

    Args:
      - 1st parameter name 
      - 2nd parameter name 
      - confidence level
      - number of points
  
    Returns: 
      - array of float (npoints * 2) contour points 
    """

    if self.iminuit_version <'2':
      print("!!! getContour not implemented vor iminuit vers.<2")
      return
    return self.minuit.mncontour(pnam1, pnam2, cl=cl, size=npoints)


  def plot_Profile(self, pnam, range=2., npoints=30):
    """plot profile likelihood of parameter pnam

    Args:
      - parameter name
      - scan range in sigma, are tuple with lower and upper value
      - number of points

    Returns: 
      - matplotlib figure 
    """

    fig = plt.figure(num='Likelihood profile ' + pnam,
                     figsize=(5., 5.))
    self.minuit.draw_mnprofile(pnam, bound=range, size=npoints, subtract_min=True)
    
    if (self.ErrDef == 1):
      ylabel='$\Delta (-2\ln\cal{L})$'
    else:
      ylabel='$\Delta (-\ln\cal{L})$'
    plt.ylabel(ylabel)
    return fig


  def plot_clContour(self, pnam1, pnam2, cl):
    """plot a contour of parameters pnam1 and pnam2
    with confidence level(s) cl
    """
    if self.iminuit_version <'2':
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
    if self.iminuit_version <'2':
      self.minuit.draw_mncontour(pnam1, pnam2, nsigma=nsig)
    else:
      ns = range(1, nsig+1)
      dchi2 = np.array(ns)**2
      cl = self.Chi22CL(dchi2)    
      self.minuit.draw_mncontour(pnam1, pnam2, cl=cl)    
    return fig

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

if __name__ == "__main__": # --- interface and example
  
  def example_xyFit():
  #
  # *** Example of an application of phyFit.xyFit()
  #
  # define the model function to fit
    def exp_model(x, A=1., x0=1.):
      return A*np.exp(-x/x0)

    # another model function
    def poly2_model(x, a=0.1, b=1., c=1.):
      return a*x**2 + b*x + c

    # set model to use in fit
    fitmodel=exp_model  # also try poly2_model !
  
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

    # perform fit to data with function xyFit using class mnFit
    resultDict = xyFit(fitmodel, data_x, data_y,
                       sx=sabsx,
                       sy=sabsy,
                       srelx=srelx,
                       srely=srely,
                       xabscor=cabsx,
                       xrelcor=crelx,
                       yabscor=cabsy,
                       yrelcor=crely,
    #                  p0=(1., 0.5),     
    #                  constraints=['A', 1., 0.03],
    #                  constraints=[0, 1., 0.03] (alternative)
    #                  limits=('A', 0., None),  # parameter limits
    #                  fixPars = ['A'],         # fix parameter(s) 
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
    pvals, perrs, cor, chi2, pnams = resultDict.values()
    print('\n*==* xyFit Result:')
    print(" parameter names:       ", pnams)
    print(" chi2: {:.3g}".format(chi2))
    print(" parameter values:      ", pvals)
    print(" neg. parameter errors: ", perrs[:,0])
    print(" pos. parameter errors: ", perrs[:,1])
    print(" correlations : \n", cor)


  def example_indexedFit():
  #
  # *** Example of an application of phyFit.xFit() to fit indexed data
  #
  #     Coordinates in r-phi are averaged and transformed to cartesian 
    
    def cartesian_to_polar(data, x=1., y=1.):
      # determine polar coordinats from cartesian (x,y)
      nm = len(data)//2 # expect two arrays with measurements
      r = np.sqrt(x*x + y*y) * np.ones(nm)
      phi = np.arctan2(y, x) * np.ones(nm)
      return np.concatenate( (r, phi) )

    # example: (r, phi) of two space points in polar coordinates
    pars = np.array([0.9, 0.87, 0.755, 0.790])
    puncs = np.array([0.027, 0.023, 0.16, 0.13])

    # perform fit to data with function xFit using class mnFit
    resultDict = xFit(cartesian_to_polar, pars, s = puncs,
                       srel=None, sabscor=None, srelcor=None,
                       names=['r', 'r', r'$\varphi$', r'$\varphi$'],
                      # p0=(1., 1.),     
                       use_negLogL=True,
                       plot=True,
                       plot_band=True,
                       plot_cor=True,
                       showplots=False,
                       quiet=False,
                       axis_labels=['Index', 'x   \  x(*par)'], 
                       data_legend = 'Polar Data',    
                       model_legend = 'r-phi from x-y')
    plt.suptitle("mnFit example: fit to indexed data",
               size='xx-large', color='darkblue')
    
  # Print results 
    pvals, perrs, cor, chi2, pnams = resultDict.values()
    print('\n*==* xyFit Result:')
    print(" parameter names:       ", pnams)
    print(" chi2: {:.3g}".format(chi2))
    print(" parameter values:      ", pvals)
    print(" neg. parameter errors: ", perrs[:,0])
    print(" pos. parameter errors: ", perrs[:,1])
    print(" correlations : \n", cor)
    
  def example_histogramFit():
  #
  # *** Histogram Fit: Example of an application of phyFit.hFit() 
  #

  #    # define the model function to fit
    def SplusB_model(x, mu = 6.987, sigma = 0.5, a=0., s = 0.2):
      '''pdf of a Gaussian signal on top of flat background
      '''
      normal = np.exp(-0.5*((x-mu)/sigma)**2)/np.sqrt(2.*np.pi*sigma**2)
      linear = (a*(xmx+xmn)/2 + 1.)/(xmx-xmn) 
      return s * normal + (1-s) * linear 

    nbins=40
    xmn = 1
    xmx = 10
    bedges=np.linspace(xmn, xmx, nbins+1)
    bcontents = np.array([1, 1, 1, 2, 2, 2, 4, 1, 0, 3, 1, 1, 0,
                        2, 3, 3, 1, 1, 0, 2, 3, 2, 3, 1, 1, 8,
                        6, 7, 9, 1, 0, 1, 2, 1, 3, 2, 1, 3, 2, 4])
    #  
    # ---  perform fit  
    #
    resultDict = hFit(SplusB_model,
          bcontents, bedges,  # bin entries and bin edges
          p0=None,            # initial guess for parameter values 
     #     constraints=['s', val , err ],   # constraints within errors
          fixPars = ['a'],         # fix parameter(s) 
          limits=('s', 0., None),  # limits
          use_GaussApprox=False,   # Gaussian approximation
          fit_density = True,      # fit density
          plot=True,           # plot data and model
          plot_band=True,      # plot model confidence-band
          plot_cor=True,      # plot profiles likelihood and contours
          showplots=False,     # show / don't show plots
          quiet=False,         # suppress informative printout
          axis_labels=['x', 'y   \  f(x, *par)'], 
          data_legend = 'random data',    
          model_legend = 'signal + background model' )

    plt.suptitle("mnFit example: fit to histogram data",
              size='xx-large', color='darkblue')

    # Print results 
    print('\n*==* histogram fit Result:')
    pvals, perrs, cor, gof, pnams = resultDict.values()
    print(" parameter names:       ", pnams)
    print(" goodness-of-fit: {:.3g}".format(gof))
    print(" parameter values:      ", pvals)
    print(" neg. parameter errors: ", perrs[:,0])
    print(" pos. parameter errors: ", perrs[:,1])
    print(" correlations : \n", cor)

  def example_userFit():
    """**unbinned ML fit** with user-defined cost function

    This code illustrates usage of the wrapper function mFit()  
    for  class **mnFit** user-defined cost function
    """  

    # generate Gaussian-distributed data
    mu0=2.
    sig0=0.5
    np.random.seed(314159)  # initialize random generator
    data = mu0 + sig0 * np.random.randn(100)

    # define cost function: 2 * negative log likelihood of Gauß;
    def myCost(mu=1., sigma=1.):
      # simple -2*log-likelihood of a 1-d Gauss distribution
      r= (data-mu)/sigma
      return np.sum( r*r + 2.*np.log(sigma))

    resultDict = mFit(myCost,
          p0=None,                 # initial guess for parameter values 
          constraints=[['mu', 2., 0.01]], # Gaussian parameter constraints
        #  limits=('sigma', None, None),  #limits
        #  fixPars = ['mu'],        # fix parameter(s) 
          neg2logL = True,         # cost is -2 * ln(L)
          plot_cor=True,           # plot profiles likelihood and contours
          showplots=False,         # show / don't show plots
          quiet=False,              # suppress informative printout
          )

    plt.suptitle("Maximum-likelihood fit: profiles and contours",
                     size='xx-large', color='darkblue')
    # Print results
    pvals, perrs, cor, gof, pnams = resultDict.values()
    print('\n*==* user-defined cost: Fit Result:')
    print(" parameter names:       ", pnams)
    print(" parameter values:      ", pvals)
    print(" neg. parameter errors: ", perrs[:,0])
    print(" pos. parameter errors: ", perrs[:,1])
    print(" correlations : \n", cor)  

  def example_unbinnedMLFit():
    """**unbinned ML fit** of pdf to unbinned data
    
    real data from measurement with a Water Cherenkov detector ("Kamiokanne")

    numbers represent time differences (in µs) between the passage of a muon
    and the registration of a second pulse, often caused by an electron from
    the decay of the stopped muon. As such events are rare, histogramming the
    data prior to fitting would introduce shifts and biases, and therefore the
    unbinned fit is the optimal method for this and simular use cases. 
    """
    dT=[7.42, 3.773, 5.968, 4.924, 1.468, 4.664, 1.745, 2.144, 3.836, 3.132,
        1.568, 2.352, 2.132, 9.381, 1.484, 1.181, 5.004, 3.06,  4.582, 2.076,
        1.88,  1.337, 3.092, 2.265, 1.208, 2.753, 4.457, 3.499, 8.192, 5.101,
        1.572, 5.152, 4.181, 3.52,  1.344, 10.29, 1.152, 2.348, 2.228, 2.172,
        7.448, 1.108, 4.344, 2.042, 5.088, 1.02,  1.051, 1.987, 1.935, 3.773,
        4.092, 1.628, 1.688, 4.502, 4.687, 6.755, 2.56,  1.208, 2.649, 1.012,
        1.73,  2.164, 1.728, 4.646, 2.916, 1.101, 2.54,  1.02,  1.176, 4.716,
        9.671, 1.692, 9.292, 10.72, 2.164, 2.084, 2.616, 1.584, 5.236, 3.663,
        3.624, 1.051, 1.544, 1.496, 1.883, 1.92,  5.968, 5.89,  2.896, 2.76,
        1.475, 2.644, 3.6,   5.324, 8.361, 3.052, 7.703, 3.83,  1.444, 1.343,
        4.736, 8.7,   6.192, 5.796, 1.4,   3.392, 7.808, 6.344, 1.884, 2.332,
        1.76,  4.344, 2.988, 7.44,  5.804, 9.5,   9.904, 3.196, 3.012, 6.056,
        6.328, 9.064, 3.068, 9.352, 1.936, 1.08,  1.984, 1.792, 9.384, 10.15,
        4.756, 1.52,  3.912, 1.712, 10.57, 5.304, 2.968, 9.632, 7.116, 1.212,
        8.532, 3.000, 4.792, 2.512, 1.352, 2.168, 4.344, 1.316, 1.468, 1.152,
        6.024, 3.272, 4.96, 10.16,  2.14,  2.856, 10.01, 1.232, 2.668, 9.176 ]

    def exponentialDecayPDF(t, tau=2., fbg=0.2, a=1., b=11.5):
      """Probability density function 

      for an exponential decay with flat background. The pdf is normed for 
      the interval [a=1µs,  b=11.5µs); these parameters a and b must be 
      fixed in the fit! 

      :param t: decay time
      :param fbg: background
      :param tau: expected mean of the decay time
      :param a: the minimum decay time which can be measured
      :param b: the maximum decay time which can be measured
      :return: probability for decay time x
      """
      pdf1 = np.exp(-t / tau) / tau / (np.exp(-a / tau) - np.exp(-b / tau))
      pdf2 = 1. / (b - a)
      return (1 - fbg) * pdf1 + fbg * pdf2

    resultDict = mFit( exponentialDecayPDF,
          data = dT, # data - if not None, a normalised PDF is assumed as model  
          p0=None,                 # initial guess for parameter values 
        #  constraints=[['tau', 2.2, 0.01], # Gaussian parameter constraints
          limits=('fbg', 0., 1.),  # parameter limits
          fixPars = ['a', 'b'],    # fix parameter(s) 
          neg2logL = True,         # use  -2 * ln(L)
          plot=True,               # plot data and model
          plot_band=True,          # plot model confidence-band
          plot_cor=False,          # plot profiles likelihood and contours
          showplots=False,         # show / don't show plots
          quiet=False,             # suppress informative printout if True
          axis_labels=['life time  ' + '$\Delta$t ($\mu$s)', 
                       'Probability Density  pdf($\Delta$t; *p)'], 
          data_legend = '$\mu$ lifetime data',    
          model_legend = 'exponential decay + flat background' )
          
    plt.suptitle("Unbinned ML fit of an exponential + flat distribution",
                     size='xx-large', color='darkblue')
    # Print results
    pvals, perrs, cor, gof, pnams = resultDict.values()
    print('\n*==* unbinned ML Fit Result:')
    print(" parameter names:       ", pnams)
    print(" parameter values:      ", pvals)
    print(" neg. parameter errors: ", perrs[:,0])
    print(" pos. parameter errors: ", perrs[:,1])
    print(" correlations : \n", cor)  

  #
  # -------------------------------------------------------------------------
  #
  # --- run above examples

  print("*** xy fit example")
  example_xyFit()

  print("*** indexed fit example")
  example_indexedFit()

  print("\n\n*** histogram fit example")
  example_histogramFit()

  print("\n\n*** minuit fit with external cost function")
  example_userFit()

  print("\n\n*** ML minuit Fit")
  example_unbinnedMLFit()

  # show all figures
  plt.show()
