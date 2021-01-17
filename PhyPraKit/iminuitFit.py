"""package iminuitFit.py
  
  Fitting with `iminiut` (https://iminuit.readthedocs.io/en/stable/)

  This class `iminuitFit.py` uses iminuit for fitting a model 
  to data with independent and/or correlated absolute and/or 
  relative uncertainties in the x- and y-directions. 
   
  A user-defined cost function in `iminuit` with uncertainties 
  that depend on model parameters is dynamically updated during 
  the fitting process. Data points with relative errors can thus
  be referred to the model instead of the data. The derivative
  of the model function w.r.t. x is used to project the 
  covariance matrix of x-uncertainties on the y-axis. 

  The implementation in this example is minimalistic and
  intended to illustrate the principle of an advanced usage
  of `iminuit`. It is also meant to stimulate own studies with 
  special, user-defined cost functions.

  The main features of this package are:
  - definition of a custom cost function 
  - implementation of the least squares method with correlated errors
  - support for correlated x-uncertainties by projection on the y-axis
  - support of relative errors with reference to the model values  
  - evaluation of profile likelihoods to determine asymetric uncertainties
  - plotting of profile likeliood and confidence contours

  supports iminuit vers. < 2.0 and >= 2.0

  A fully functional example is provided by the function `mFit()`
  in PhyPraKit and the Python script `examples/test_mFit.py`

.. moduleauthor:: Guenter Quast <g.quast@kit.edu>
"""

import sys
import numpy as np, matplotlib.pyplot as plt 
from iminuit import __version__, Minuit
from inspect import signature, Parameter  

def build_CovarianceMatrix(nd, e=None, erel=None,
                           eabscor=None, erelcor=None, data=None):
  """
  Build a covariance matrix from independent and correlated 
  absolute and/or relative error components

  Correlated absolute and/or relative uncertainties enter in the diagonal 
  and off-diagonal elements of the covariance matrix. Covariance matrix
  elements of the individual components are added to form the complete
  Covariance Matrix.

  Args:

  * nd: number of data points
  * e: scalar, array of float, or 2d-array of float: 
    independent uncertainties or a full covariance matrix
  * erel: scalar, array of float, 2d-array of float: 
    independent relative uncertainties or a full covariance matrix
  * eabscor: float or array of float of list of arrays of float:
    absolute correlated uncertainties
  * erelcor: float or array of float of list of arrays of float:
    relative correlated uncertainties
  * data: array of float: data, needed (only) for relative uncertainties

  Returns:

  * np-array of float: covariance matrix 

  """

  # 1. independent errors
  if e is not None:
    e_ = np.asarray(e)
    if e_.ndim == 2: # already got a matrix, take as covariance matrix
      cov = e_
    else:
      e_ = np.ones(nd)*np.array(e) # ensure array of length nd
      cov = np.diag(e_*e_) # set diagonal elements of covariance matrix
  else:
    cov = np.zeros( (nd, nd) )
    
  # 2. add relative errors
  if erel is not None:
    er_ = np.asarray(erel)
    if er_.ndim == 2: # already got a matrix, take as covariance matrix
      cov += er_ * np.outer(nd, nd)
    else:
      er_ = np.array(erel) * data  # ensure array of length nd
      cov += np.diag(er_*er_)      # diagonal elements of covariance matrix
        
  # 3. add absolute, correlated error components  
  if eabscor is not None:
    eac=np.asarray(eabscor)
    if len(np.shape(eac )) < 2: # has one entry
      c_ = eac * np.ones(nd)
      cov += np.outer(c_, c_) 
    else:            # got a list, add each component
      for c in eabscor:
        c_ = np.array(c)*np.ones(nd)
        cov += np.outer(c_, c_)
        
  # 4. add relative, correlated error components
  if erelcor is not None:
    ear=np.asarray(erelcor)
    if len(np.shape(ear) ) < 2: # has one entry
      c_ = ear * data
      cov += np.outer(c_, c_) 
    else:            # got a list, add each component
      for c in erelcor:
        c_ = np.array(c) * data
        cov += np.outer(c_, c_) 
  # return complete matrix
  return cov

class iminuitFit():
  """**Fit an arbitrary funtion f(x, *par) to data**  
  with independent and/or correlated absolute and/or relative uncertainties
   
  Public methods:

  - init_data():        initialze data and uncertainties
  - init_fit():         initialize fit: data, model and parameter constraints
  - setOptions():       set options
  - do_fit():           perform fit
  - plotModel():        plot model function and data
  - plotContours():     plot profile likelihoods and confidence contours 
  - getResult():        access to results 
 
  Public data members:

  - ParameterNames:     names of parameters (as specified in model function)
  - Chi2:               chi2 at best-fit point
  - NDoF:               number of degrees of freedom
  - ParameterValues:    parameter values at best-fit point
  - MigradErrors:       symmetric uncertainties
  - CovarianceMatrix:   covariance matrix
  - CorrelationMatrix:  correlation matrix
  - OneSigInterval:     one-sigma (68% CL) ranges of parameer values 
 
  - covx:     covariance matrix of x-data
  - covy:     covariance matrix of y-data 
  - cov:      combined covariance matrix, including projected x-uncertainties

  """

  def __init__(self):

    # no data or model provided yet
    self.data = None
    self.costf = None
    # no fit done
    self.migradResult = None
    self.minosResult = None
    # default options
    self.refModel=True
    self.run_minos = True
    self.quiet = True
    
  def setOptions(self,
              relative_refers_to_model=None,
              run_minos=None,
              quiet=None):
    # define options for fit
    #   - rel. errors refer to model else data
    #   - run minos else don*t run minos 
    #   - don*t provide printout else verbose printout 

    if relative_refers_to_model is not None:
      self.refModel = relative_refers_to_model
    if run_minos is not None:   
      self.run_minos = run_minos
    if quiet is not None:
      self.quiet = quiet
    
  def init_data(self,
                x, y,             
                ex=None, ey=1.,
                erelx=None, erely=None,
                cabsx=None, crelx=None,
                cabsy=None, crely=None):

    # create data object
    self.data = self.DataUncertainties(x, y, ex, ey,
                    erelx, erely, cabsx, crelx, cabsy, crely,
                    quiet=self.quiet)
    
    # set flags for steering of fit process in do_fit()
    self.iterateFit = self.data.has_xErrors or(
         self.data.has_rel_yErrors and self.refModel)

  def init_fit(self, model, p0=None, constraints=None):
    # set model function
    self.model = model
    # create cost function
    self.costf = self.LSQwithCov(self.data, self.model, quiet=self.quiet)
    if constraints is not None:
      self.costf.setConstraints(constraints)

    # inspect parameters of model function to set start values for fit
    sig=signature(self.model)
    parnames=list(sig.parameters.keys())
    ipardict={}
    if p0 is not None:
      for i, pnam in enumerate(parnames[1:]):
        ipardict[pnam] = p0[i]
    else:
    # try defaults of parameters from argument list
      for i, pnam in enumerate(parnames[1:]):
        dv = sig.parameters[pnam].default   
        if dv is not Parameter.empty:
          ipardict[pnam] = dv
        else:
          ipardict[pnam] = 0.  #  use zero in worst case

    # create Minuit object
    if __version__ < '2':
      if self.quiet:
        print_level=0
      self.minuit = Minuit(self.costf, **ipardict,
                           errordef=1., print_level=print_level)
    else:
      self.minuit = Minuit(self.costf, **ipardict)  
      self.minuit.errordef = 1.
      if self.quiet:
        self.minuit.print_level = 0.

  def do_fit(self):

    if self.data is None:
      print(' !!! iminuitFit: no data object defined - call init_data()')
      sys.exit('iminuitFit Error: no data object')
    if self.costf is None:
      print(' !!! iminuitFit: no fit object defined - call init_fit()')
      sys.exit('iminuitFit Error: no fit object')
    
    # perform initial fit if everything ok
    self.migradResult = self.minuit.migrad()  # find minimum of cost function
    # possibly, need to iterate fit
    if self.iterateFit:
      if not self.quiet:
        print( '*==* iminuitFit iterating ',
               'to take into account parameter-dependent uncertainties')

      # enable dynamic calculation of covariance matrix
      self.data._set_dynamicCovMat(self.refModel, self.costf.model)
      # fit with dynamic recalculation of covariance matrix
      self.migradResult = self.minuit.migrad()

    # run profile likelihood scan to check for asymmetric errors
    if self.run_minos:
      self.minosResult = self.minuit.minos()

    self._storeResult()
    
    return self.migradResult, self.minosResult
  
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
      
    if self.minosResult is not None:
      pmerrs = [] 
    #  print("MINOS errors:")
      if __version__< '2':
        for pnam in m.merrors.keys():
          pmerrs.append([m.merrors[pnam][2], m.merrors[pnam][3]])
     #    print(f"{3*' '}{pnam}: {m.merrors[pnam][2]:.2g}",
     #                       f"+{m.merrors[pnam][3]:.2g}")
      else:
        for pnam in m.merrors.keys():
          pmerrs.append([m.merrors[pnam].lower, m.merrors[pnam].upper])
      #  print(f"{3*' '}{pnam}: {m.merrors[pnam].lower:.2g}",
      #                      f"+{m.merrors[pnam].upper:.2g}")      
      self.OneSigInterval=np.array(pmerrs)
    else:
      self.OneSigInterval = np.array(list(zip(-parerrs, parerrs)))

    # call cost function at miminum to update all parameters
    fval = self.costf(*parvals) 
  
    # store results as class members
    #   parameter names
    self.ParameterNames = parnames
    #   chi2 at best-fit point (possibly different from minCost)
    self.Chi2 = self.costf.chi2  
    #   parameter values at best-fit point
    self.ParameterValues = np.array(parvals, copy=True)
    #   number of degrees of freedom
    self.NDoF = ndof  
    #   symmetric uncertainties
    self.MigradErrors = np.array(parerrs, copy=True)
    #   covariance and correlation matrices
    self.CovarianceMatrix = np.array(cov, copy=True)
    self.CorrelationMatrix = cov/np.outer(parerrs, parerrs)
    #   1-sigma (68% CL) range in 
    # self.OneSigInterval
    
  def getResult(self):
    # return most im portant results
    return (self.ParameterValues, self.OneSigInterval,
            self.CorrelationMatrix, self.Chi2)

  class DataUncertainties:
    """
    Handle data and uncertainties

    Args:

    -  x:       abscissa - "x values"
    -  y:       ordinate - "y values"
    -  ex:      independent uncertainties x
    -  ey:      independent uncertainties y
    -  erelx:   independent relative uncertainties x
    -  erely:   independent relative uncertainties y
    -  cabsx:   correlated abolute uncertainties x
    -  crelx:   correlated relative uncertainties x
    -  cabsy:   correlated absolute uncertainties y
    -  crely:   correlated relative uncertainties y
    -  quiet:   no informative printout if True

    Data members:
  
      * copy of all input arguments
      * covx: covariance matrix of x
      * covy: covariance matrix of y uncertainties
      * cov: full covariance matrix incl. projected x
      * iCov: inverse of covariance matrix

    """
    
    def __init__(self, x, y, 
                 ex, ey, erelx, erely, cabsx, crelx, cabsy, crely,
                 quiet=True):
      self.x = np.asarray(x) # abscissa - "x values"
      self.y = np.asarray(y) # ordinate - "y values"
      self.ex = ex            # independent uncertainties x
      self.ey = ey            # independent uncertainties y
      self.erelx = erelx      # independent relative uncertainties x
      self.erely = erely      # independent relative uncertainties y
      self.cabsx = cabsx      # correlated abolute uncertainties x
      self.crelx = crelx      # correlated relative uncertainties x
      self.cabsy = cabsy      # correlated absolute uncertainties y
      self.crely = crely      # correlated relative uncertainties y
      self.quiet = quiet      # no informative printout if True

      self.nd = len(x) 
      self.model = None # no model defined yet

      # set flags for steering of fit process in do_fit()
      self.rebulildCov = None
      self.has_xErrors = ex is not None or erelx is not None \
        or cabx is not None or crelx is not None
      self.has_rel_yErrors = erely is not None or crely is not None
      self.needs_covariance = \
        self.cabsx is not None or self.crelx is not None or \
        self.cabsy is not None or self.crely is not None 
            
      # build (initial) covariance matrix (ignore x-errors)
      cov_initial = build_CovarianceMatrix(self.nd, ey, erely, cabsy, crely, y)

      # initialize uncertainties and eventually covariance matrix
      self.initialCov(cov_initial)
      
    def initialCov(self, err):
      """Build initial (static) covariance matrix (for pre-fit)
      and calculate inverse matrix
      """
      self.err2 = np.asarray(err)
      if self.err2.ndim == 2:
       # got a covariance matrix, need inverse
        self.needs_Covariance = True
        self.iCov = np.matrix(self.err2).I
      else:
        self.err2 *= self.err2 
        self.iCov = 1./self.err2
      # do not trigger rebuild of covariance matrix in cost function
      self.rebuildCov = False 

      # store matrix components
      self.covy = self.err2
      self.covx = None
      self.cov = self.covy      
      
    def _set_dynamicCovMat(self, ref_toModel = False, model = None):
      # method to switch on dynamic re-calculation of covariance matrix 
      self.ref_toModel = ref_toModel
      self.model = model

      # rebuild covariance matrix during fitting procedure
      self.rebuildCov = True        # flag used in cost function
      self.needs_Covariance = True  # use matrix in cost function
          # !!! might achieve speed-up for simple problems
          #  if also implement version w.o. covariance matrix

      # build static (=parameter-independent) part of y-covariance Matrix
      self.nd = len(self.x)
      if self.has_rel_yErrors and self.ref_toModel:
        # only some y-errors are parameter-independent
        self._staticCov = build_CovarianceMatrix(self.nd,
                       self.ey, eabscor = self.cabsy)
      else: 
        # all y-errors are parameter-independent
        self._staticCov = build_CovarianceMatrix(self.nd,
                                  self.ey, erel=self.erely,
                                  eabscor = self.cabsy, erelcor=self.crely,
                                                data=self.y)

      if self.ref_toModel and self.has_rel_yErrors:
        # build matrix of relative errors
        self._covy0 = build_CovarianceMatrix(self.nd,
                                  erel=self.erely,
                                  erelcor=self.crely,
                                  data=np.ones(self.nd))
      else:
        self._covy0 = None
        
      # covariance matrix of x-uncertainties (all are parameter-dependent)
      if self.has_xErrors:
        self.covx = build_CovarianceMatrix(self.nd,
                              self.ex, self.erelx,
                              self.cabsx, self.crelx,
                              self.x)
       #  determine dx for derivative from smallest x-uncertainty
        self._dx = np.sqrt(min(np.diag(self.covx)))/10.
      else:
        self.covx = None

    def _rebuild_Cov(self, mpar):
      """
      (Re-)Build the covariance matrix from components
      """
      # start from pre-built parameter-independent part of Covariance Matrix
      self.cov = np.array(self._staticCov, copy=True)

      # add matrix of parameter-dependent y-uncertainties
#      if self.ref_toModel and self.has_rel_yErrors:
      if self._covy0 is not None:
        ydat = self.model(self.x, *mpar)       
        self.cov += self._covy0 * np.outer(ydat, ydat)
      # store covariance matrix of y-uncertainties    
      self.covy = np.array(self.cov, copy=True)

      # add projected x errors
      if self.has_xErrors:
       # determine derivatives of model function w.r.t. x,
        mprime = 0.5 / self._dx * (
                 self.model(self.x + self._dx,*mpar) - 
                 self.model(self.x - self._dx,*mpar) )
        # project on y and add to covariance matrix
        self.cov += np.outer(mprime, mprime) * self.covx

      # inverse covariance matrix 
      self.iCov = np.matrix(self.cov).I

    def getCov(self):
      """return covariance matrix of data
      """
      return self.cov
  
    def getxCov(self):
      """return covariance matrix of x-data
      """
      return self.covx

    def getyCov(self):
      """return covariance matrix of y-data
      """
      return self.covy
    
    def getiCov(self):
      """return inverse of covariance matrix, as used in cost function
      """
      return self.iCov
      
  # define custom cost function for iminuit
  class LSQwithCov:
    """
    custom Least-SQuares cost function with error matrix

    Input:

    - data object of type DataUncertainties
    - model function f(x, \*par)
    - use_neglogL: use full 2*neg Log Likelihood instead of chi2 if True

    __call__ method of this class is called by iminuit

    Data members:

    - ndof: degrees of freedom 
    - nconstraints- number of constraints
    - chi2: chi2-value (goodness of fit)
    - use_2neglogL: usage of full 2*neg Log Likelihood
    - quiet: no printpout if True    

    """
  
    def __init__(self, data, model, quiet=True, use_negLogL = False):
      from iminuit.util import describe, make_func_code

      self.data = data
      self.model = model
      self.quiet = quiet
      # use -2 * log(L) of Gaussian instead of Chi2
      #  (only different from Chi2 for parameter-dependent uncertainties)
      self.use_2negLogL = use_negLogL
      
      # set proper signature of model function for iminuit
      self.pnams = describe(model)[1:]
      self.func_code = make_func_code(self.pnams)
      self.npar = len(self.pnams)
      # dictionary assigning parameter name to index
      self.pnam2id = {
        self.pnams[i] : i for i in range(0,self.npar)
        } 
      self.ndof = len(data.y) - self.npar
      self.nconstraints = 0

    def setConstraints(self, constraints):
      """Add parameter constraints

      format: list or list of lists of type 
      [parameter name, value, uncertainty] or
      [parameter index, value, uncertainty]
      """
      
      if isinstance(constraints[1], list):
         self.constraints = constraints
      else:
         self.constraints = [constraints]
      self.nconstraints = len(self.constraints)
      # take account of constraints in degrees of freedom 
      self.ndof = len(self.data.y) - self.npar + self.nconstraints
          
    def __call__(self, *par):  
      # called iteratively by minuit

      # cost funtion is standard chi2;
      #   full -2 * log L of a Gaussian distribution optionally
      
      nlL2 = 0. # initialize -2*log(L)
      #  first, take into account possible parameter constraints  
      if self.nconstraints:
        for c in self.constraints:
          if type(c[0])==type(' '):
            p_id = self.pnam2id[c[0]]
          else:
            p_id = c[0]
          r = ( par[p_id] - c[1]) / c[2] 
          nlL2 += r*r

      # check if Covariance matrix needs rebuilding
      if self.data.rebuildCov:
        self.data._rebuild_Cov(par)

      # add chi2 of data wrt. model    
      resid = self.data.y - self.model(self.data.x, *par)
      if not self.data.needs_Covariance:
        # fast calculation for simple errors
        nlL2 += np.sum(resid * self.data.iCov*resid)
        # identical to classical Chi2
        self.chi2 = nlL2
        if self.use_2negLogL:
           # take into account parameter-dependent normalisation term
          nlL2 += np.log(np.prod(self.data.err2))
      else:
        # with full inverse covariance matrix for correlated errors
        nlL2 += float(np.inner(np.matmul(resid.T, self.data.iCov), resid))
        #  up to here, identical to classical Chi2
        self.chi2 = nlL2
        if self.use_2negLogL:
         # take into account parameter-dependent normalisation term
          nlL2 += np.log(np.linalg.det(self.data.cov))

      return nlL2
    
  # --- end definition of class LSQwithCov ----

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
    dp = 0.01 * np.sqrt(np.diag(covp))
    #   derivative df/dp_j at each x_i
    dfdp = np.empty( (len(pvals), len(x)) )
    for j in range(len(pvals)): 
      p_plus = np.array(pvals, copy=True)
      p_plus[j] = pvals[j] + dp[j]
      p_minus = np.array(pvals, copy=True)
      p_minus[j] = pvals[j] - dp[j]
      dfdp[j] = 0.5 / dp[j] * (
                    model(x, *p_plus) - 
                    model(x, *p_minus) )
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
    
    Uses iminuitObject abd cost Fuction of type LSQwithCov

    Args: 
      * list of str: axis labels
      * str: legend for data
      * str: legend for model 

    Returns:
      * matplotlib figure
    """
    
  # access low-level fit objects
    m = self.minuit  # minos object
    cf = self.costf  # cost function object

  # retrieve fit results
    pvals, pmerrs, cor, chi2 = self.getResult()
    pcov = self.CovarianceMatrix
    pnams = self.ParameterNames
    ndof = cf.ndof

  # get data
    x = cf.data.x
    y = cf.data.y
    ey = cf.data.getyCov()
    if ey.ndim ==2:
      ey = np.sqrt(np.diag(ey))
    else:
      ey = np.sqrt(ey)
    ex = cf.data.getxCov()
    if ex is not None:
      if ex.ndim ==2:
        ex = np.sqrt(np.diag(ex))
      else:
        ex = np.sqrt(ex)

   # draw data and fitted line
    fig_model = plt.figure(figsize=(7.5, 6.5))
    if ex is not None:
      plt.errorbar(x, y, xerr=ex, yerr=ey, fmt='x', label=data_legend)
    else:
      plt.errorbar(x, y, ey, fmt="x", label='data')
    xmin, xmax = plt.xlim()
    xplt=np.linspace(xmin, xmax, 100)
    yplt = cf.model(xplt, *pvals)
    plt.plot(xplt, yplt, label=model_legend)
    plt.xlabel(axis_labels[0], size='x-large')
    plt.ylabel(axis_labels[1], size='x-large')
    plt.grid()
   # draw error band around model function
    if plot_band:
      DeltaF = self.getFunctionError(xplt, cf.model, pvals, pcov)
      plt.fill_between(xplt, yplt+DeltaF, yplt-DeltaF, alpha=0.1,
                           color='green')    

    # display legend with some fit info
    fit_info = [
    f"$\\chi^2$ / $n_\\mathrm{{dof}}$ = {chi2:.1f} / {ndof}",]
    if self.minosResult is not None:
      for p, v, e in zip(pnams, pvals, pmerrs):
        fit_info.append(f"{p} = ${v:.3f}^{{+{e[1]:.2g}}}_{{{e[0]:.2g}}}$")
    else:
     for p, v, e in zip(pnams, pvals, perrs):
        fit_info.append(f"{p} = ${v:.3f}\pm{{{e:.2g}}}$")
    plt.legend(title="\n".join(fit_info))      

    return fig_model
  
# plot array of profiles and contours
  def plotContours(self):
    """
    Plot grid of profile curves and one- and two-sigma
    contours lines from iminuit object

    Arg: 
      * iminuitObject

    Returns:
      * matplotlib figure 
    """

    def CL2Chi2(CL):
      '''
      calculate DeltaChi2 from confidence level CL
      '''
      return -2.*np.log(1.-CL)

    def Chi22CL(dc2):
     '''
     calculate confidence level CL from DeltaChi2
     '''
     return (1. - np.exp(-dc2 / 2.))


    m = self.minuit     
    npar = m.nfit    # numer of parameters
    if __version__< '2':
      pnams = m.values.keys()  # parameter names
    else:
  # vers. >=2.0 
      pnams = m.parameters      # parameter names

    fsize=3.5
    cor_fig, axarr = plt.subplots(npar, npar,
                                figsize=(fsize*npar, fsize*npar))
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
              cl=(Chi22CL(1.), Chi22CL(4.)) )
    return cor_fig 

if __name__ == "__main__": # --- interface and example
  def mFit(fitf, x, y, sx = None, sy = None,
         srelx = None, srely = None, 
         xabscor = None, xrelcor = None,        
         yabscor = None, yrelcor = None,
         p0 = None, constraints = None, 
         plot = True, plot_cor = True,
         plot_band=True, quiet = False,
         axis_labels=['x', 'y = f(x, *par)'], 
         data_legend = 'data',    
         model_legend = 'model'): 
    """
    fit an arbitrary function f(x) to data
    with uncorrelated and correlated absolute and/or relative errors on y 
    with package iminuit

    Args:
      * fitf: model function to fit, arguments (float:x, float: *args)
      * x:  np-array, independent data
      * y:  np-array, dependent data
      * sx: scalar or 1d or 2d np-array , uncertainties on x data
      * sy: scalar or 1d or 2d np-array , uncertainties on x data
      * srelx: scalar or np-array, relative uncertainties x
      * srely: scalar or np-array, relative uncertainties y
      * yabscor: scalar or np-array, absolute, correlated error(s) on y
      * yrelcor: scalar or np-array, relative, correlated error(s) on y
      * p0: array-like, initial guess of parameters
      * constraints: list or list of lists with [name or id, value, error]
      * plot: show data and model if True
      * plot_cor: show profile liklihoods and conficence contours
      * plot_band: plot uncertainty band around model function
      * quiet: suppress printout
      * list of str: axis labels
      * str: legend for data
      * str: legend for model 

    Returns:
      * np-array of float: parameter values
      * 2d np-array of float: parameter uncertaities [0]: neg. and [1]: pos. 
      * np-array: correlation matrix 
      * float: chi2  \chi-square of fit a minimum
    """

    ## from .iminuitFit import iminuitFit

    # ... check if errors are provided ...
    if sy is None:
      sy = np.ones(len(y))
      print('\n!**! No y-errors given',
            '-> parameter errors from fit are meaningless!\n')
  
    # set up a fit object
    Fit = iminuitFit()

    # set some options
    Fit.setOptions(run_minos=True, relative_refers_to_model=True)

    # pass data and uncertainties to fit object
    Fit.init_data(x, y,
                ex = sx, ey = sy,
                erelx = srelx, erely = srely,
                cabsx = xabscor, crelx = xrelcor,
                cabsy = yabscor, crely = yrelcor)

    # pass model fuction, start parameter and possibe constraints
    Fit.init_fit(fitf, p0=p0, constraints=constraints)

    # perform the fit
    fitResult = Fit.do_fit()
    # print fit resule (dictionary from migrad/minos(
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
    #   prifile likelihood scan and confidence contours
    if plot_cor:
      fig_cor = Fit.plotContours()

    # show plots on screen
    if plot or plot_cor:
      plt.show()

    # return
    #   numpy arrays with fit result: parameter values,
    #   negative and positive parameter uncertainties,
    #   correlation matrix and chi2
    return Fit.getResult()
  
# -----------------------------------------------------------------

  #
  # Example of an application of iminuitFit.mFit()
  #
  from PhyPraKit import generateXYdata

  # define the model function to fit
  def model(x, A=1., x0=1.):
    return A*np.exp(-x/x0)
  mpardict = {'A':1., 'x0':0.5}  # model parameters

# set error components 
  sabsy = 0.07
  srely = 0.05 # 5% of model value
  cabsy = 0.04
  crely = 0.03 # 3% of model value
  sabsx = 0.05
  srelx = 0.04 # 4%
  cabsx = 0.03
  crelx = 0.02 # 2%

# generate pseudo data
  np.random.seed(314)      # initialize random generator
  nd=10
  data_x = np.linspace(0, 1, nd)       # x of data points
  sigy = np.sqrt(sabsy * sabsy + (srely*model(data_x, **mpardict))**2)
  sigx = np.sqrt(sabsx * sabsx + (srelx * data_x)**2)
  xt, yt, data_y = generateXYdata(data_x, model, sigx, sigy,
                                      xabscor=cabsx,
                                      xrelcor=crelx,
                                      yabscor=cabsy,
                                      yrelcor=crely,
                                      mpar=mpardict.values() )

# perform fit to data with function mFit using iminuitFit class
  parvals, parerrs, cor, chi2 = mFit(model, data_x, data_y,
                                     sx=sabsx,
                                     sy=sabsy,
                                     srelx=srelx,
                                     srely=srely,
                                     xabscor=cabsx,
                                     xrelcor=crelx,
                                     yabscor=cabsy,
                                     yrelcor=crely,
                                     p0=(1., 0.5),
#                                     constraints=['A', 1., 0.03],
#                                     constraints=[0, 1., 0.03],
                                     plot=True,
                                     plot_band=True,
                                     plot_cor=True,
                                     quiet=False,
                                     axis_labels=['x', 'y   \  f(x, *par)'], 
                                     data_legend = 'random data',    
                                     model_legend = 'exponential model')

# Print results to illustrate how to use output
  print('\n*==* Fit Result:')
  print(f" chi2: {chi2:.3g}")
  print(f" parameter values:      ", parvals)
  print(f" neg. parameter errors: ", parerrs[:,0])
  print(f" pos. parameter errors: ", parerrs[:,1])
  print(f" correlations : \n", cor)
  
  plt.show()
