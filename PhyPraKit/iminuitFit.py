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
  """  
  Fit an arbitrary funtion f(x, *par) to data with
   independent and/or correlated absolute and/or relative untertainties
  """

  def __init__(self):
    self.options()

  def options(self,
              relative_refers_to_model=True,
              run_minos=True):
    # define options for fit
    # - ? rel. errors refer to data or model
    # - ? run minos
    self.refModel=relative_refers_to_model
    self.run_minos = run_minos
    self.quiet=True
    
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
      self.costf.addConstraints(constraints)

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
    # perform initial fit
    result = self.minuit.migrad()  # find minimum of cost function
    # possibly, need to iterate fit
    if self.iterateFit:
      if not self.quiet:
        print( '*==* mFit iterating ',
               'to take into account parameter-dependent uncertainties')

      # enable dynamic calculation of covariance matrix
      self.data.set_dynamicCovMat(self.refModel, self.costf.model)
      # fit with dynamic recalculation of covariance matrix
      result = self.minuit.migrad()

    # run profile likelihood scan to check for asymmetric errors
    if self.run_minos:
      minosResult = self.minos_err = self.minuit.minos()
    else:
      minosResult = None

    return result, minosResult
  
  def result(self):
  # report results as numpy arrays
    m=self.minuit
    # extract result parametes !!! this part depends on iminuit version !!!
    chi2 = m.fval                                   # chi2 
    npar = m.nfit                                   # numer of parameters
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
      
    if self.run_minos:
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
      pmerrs=np.array(pmerrs)
    else:
      pmerrs = np.array(list(zip(-parerrs, parerrs)))
      
    # remember results
    self.CovarianceMatrix = np.array(cov, copy=True)
    self.CorrelationMatrix = cov/np.outer(parerrs, parerrs)
    self.ParameterErrors = np.array(parerrs, copy=True)
    self.ParameterValues = np.array(parvals, copy=True)
    self.ParameterNames = parnames
    self.MinosErrors = np.array(pmerrs, copy = True)
    self.Chi2 = chi2
    self.NDoF = ndof

    #return result arrays
    return (self.ParameterValues, self.MinosErrors,
            self.CorrelationMatrix, self.Chi2)

  class DataUncertainties:
    """
    class to handle data and uncertainties
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
      if ex is not None or erelx is not None \
        or cabx is not None or crelx is not None:
        self.has_xErrors= True
      else:
        self.has_xErrors= False

      if erely is not None or crely is not None: 
        self.has_rel_yErrors = True
      else:
        self.has_rel_yErrors = False

      # build (initial) covariance matrix (ignore x-errors)
      cov_initial = build_CovarianceMatrix(self.nd, ey, erely, cabsy, crely, y)

      # initialize uncertainties and eventually covariance matrix
      self.initialCov(cov_initial)
      
    def initialCov(self, err):
      # special init to set up covariance matrix and its inverse
      self.err2 = np.asarray(err)
      self.errdim = self.err2.ndim
      if self.errdim == 2:
      # got a covariance matrix, need inverse
        self.iCov = np.matrix(self.err2).I
      else:
        self.err2 *= self.err2 
        self.iCov = 1./self.err2
      # do not trigger rebuild of coariance matrix in cost function
      self.rebuildCov = False 

      # store matrix components
      self.covy = self.err2
      self.covx = None
      self.cov = self.covy      

      
    def set_dynamicCovMat(self, ref_toModel = False, model = None):
      # method to switch on dynamic re-calculation of covariance matrix 
      self.ref_toModel = ref_toModel
      self.model = model

      # rebuild covariance matrix during fitting procedure
      self.rebuildCov = True  # flag used in cost function
      self.errdim = 2         # use full covariance matrix in cost function

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


    def rebuild_Cov(self, mpar):
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

    def get_Cov(self):
      return self.cov
  
    def get_xCov(self):
      return self.covx

    def get_yCov(self):
      return self.covy
    
    def get_iCov(self):
      return self.iCov
      
  # define custom cost function for iminuit
  class LSQwithCov:
    """
    custom Least-SQuares cost function with error matrix
    """
  
    def __init__(self, data, model, quiet=True):
      from iminuit.util import describe, make_func_code

      self.data = data
      self.model = model
      self.quiet = quiet
      
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

    def addConstraints(self, constraints):
      # add parameter constraints
      #  format: list or list of lists with [name, value, uncertainty]
      if isinstance(constraints[1], list):
         self.constraints = constraints
      else:
         self.constraints = [constraints]
      self.nconstraints = len(self.constraints)
      # take account of constraints in degrees of freedom 
      self.ndof = len(self.data.y) - self.npar + self.nconstraints
 
         
    def __call__(self, *par):  # accept a variable number of model parameters
      # called iteratively by minuit

      dc = 0. 
      #  first, take into account possible parameter constraints  
      if self.nconstraints:
        for c in self.constraints:
          if type(c[0])==type(' '):
            p_id = self.pnam2id[c[0]]
          else:
            p_id = c[0]
          r = ( par[p_id] - c[1]) / c[2] 
          dc += r*r

      # check if Covariance matrix needs rebuilding
      if self.data.rebuildCov:
        self.data.rebuild_Cov(par)

      # add chi2 of data wrt. model    
      resid = self.data.y - self.model(self.data.x, *par)
      if self.data.errdim < 2:
        # fast calculation for simple errors
        return dc + np.sum(resid * self.data.iCov*resid)
      else:
        # with full inverse covariance matrix for correlated errors
        return dc + np.inner(np.matmul(resid.T, self.data.iCov), resid)

  # --- end definition of class LSQwithCov ----

  @staticmethod
  def getFunctionError(x, model, pvals, covp):
    """ determine error of model at x  
    
    Formula: 
    Delta(x) = sqrt( sum_i,j (df/dp_i(x) df/dp_j(x) Vp_i,j) )

    Args:

      * x: scalar or np aray of x values
      * model: model function
      * pvlas: parameter values
      * covp: covariance matrix of parameters

    Returns:

      * model uncertainty, same length as x

    """

    # calculate derivatives of model w.r.t parameters    
    #    parameter step size 
    dp = 0.01 * np.sqrt(np.diag(covp))
    # derivative df/dp_j at each x_i
    Delta = np.zeros(len(x))
    for i in range(len(x)):
      dfdp = np.zeros(len(pvals))
      for j in range(len(pvals)): 
        p_plus = np.array(pvals, copy=True)
        p_plus[j] = pvals[j] + dp[j]
        p_minus = np.array(pvals, copy=True)
        p_minus[j] = pvals[j] - dp[j]
        dfdp[j] = 0.5 / dp[j] * (
                      model(x[i], *p_plus) - 
                      model(x[i], *p_minus) )
      Delta[i] = np.sum(np.outer(dfdp, dfdp) * covp)
    return np.sqrt(Delta) 
  
  def plotModel(self,
                axis_labels=['x', 'y = f(x, *par)'], 
                data_legend = 'data',    
                model_legend = 'fit',
                plot_band=True): 
    """
    Plot model function and data 

    Args: 
      * iminuitObject
      * cost Fuction of type LSQwithCov
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
    pvals, pmerrs, cor, chi2 = self.result()
    pcov = self.CovarianceMatrix
    pnams = self.ParameterNames
    ndof = cf.ndof
    
    # find out if minos ran
    if m.merrors:
       minos_done = True
    else:
       minos_done = False
 
  # get data
    x = cf.data.x
    y = cf.data.y
    ey = cf.data.get_yCov()
    if ey.ndim ==2:
      ey = np.sqrt(np.diag(ey))
    else:
      ey = np.sqrt(ey)
    ex = cf.data.get_xCov()
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

   # draw error band around model function
    if plot_band:
      DeltaF = self.getFunctionError(xplt, cf.model, pvals, pcov)
      plt.gca().fill_between(xplt, yplt+DeltaF, yplt-DeltaF, alpha=0.1,
                           color='green')    

    # display legend with some fit info
    fit_info = [
    f"$\\chi^2$ / $n_\\mathrm{{dof}}$ = {chi2:.1f} / {ndof}",]
    if minos_done:
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
    Plot grid of profile curves and one- and tow-sigma
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
