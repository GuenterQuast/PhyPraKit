#!/usr/bin/python3
# module PhyPraKit.py, python3 version
def A0_readme():
  # name chosen to make sure sphinx puts this docstring at the top
  """Package PhyPraKit

  **PhyPraKit**  for Data Handling, Visualisation and Analysis

  contains the following functions:

    1. Data input/output:

      - readColumnData() read data and meta-data from text file
      - readCSV()        read data in csv-format from file with header
      - readtxt()        read data in "txt"-format from file with header
      - readPicoScope()  read data from PicoScope
      - readCassy()      read CASSY output file in .txt format   
      - labxParser()     read CASSY output file, .labx format   
      - writeCSV()       write data in csv-format (opt. with header)
      - writeTexTable()  write data in LaTeX table format
      - round_to_error() round to same number of significant digits as uncertainty
      - ustring()        return rounded value +/- uncertainty as formatted string;   
        alternative: the data type *ufloat(v,u)* of package *uncertainties* comfortably 
        supports printing of values *v* with uncertainties *u*.

    2. signal processing:

      - offsetFilter()      subtract an offset in array a
      - meanFilter()        apply sliding average to smoothen data
      - resample()          average over n samples
      - simplePeakfinder()  find peaks and dips in an array,    
        `(recommend to use convolutionPeakfinder)`
      - convolutionPeakfinder() find maxima (peaks) in an array
      - convolutionEdgefinder() find maxima of slope (rising) edges in an array
      - Fourier_fft()       fast Fourier transformation of an array
      - FourierSpectrum()   Fourier transformation of an array   
        `(slow, preferably use fft version)`
      - autocorrelate()     auto-correlation function

    3. statistics:

      - wmean                  calculate weighted mean
      - BuildCovarianceMatrix  build covariance matrix from individual uncertainties
      - Cov2Cor                convert covariance matrix to correlation matrix
      - Cor2Cov                convert correlation matrix + errors to covariance matrix 
      - chi2prob               caclulate chi^2 probability 
      - propagatedError        determine propagated uncertainty, with covariance;  
        hint: the data type *ufloat(v,u)* of package *uncertainties* comfortably supports 
        functions of values *v* with uncertainties *u* with correct error propagation
      - getModelError          determine uncertainty of parameter-dependent model function  

    4. histograms tools:

      - barstat()   statistical information (mean, sigma, error_on_mean) from bar chart
      - nhist()    histogram plot based on np.historgram() and plt.bar()    
        `better use matplotlib.pyplot.hist()`  
      - histstat() statistical information from 1d-histogram
      - nhist2d()  2d-histotram plot based on np.histrogram2d, plt.colormesh()  
        `(better use matplotlib.pyplot.hist2d)`  
      - hist2dstat() statistical information from 2d-histogram
      - profile2d()  "profile plot" for 2d data
      - chi2p_indep2d() chi2 test on independence of data
      - plotCorrelations()  distributions and correlations of a multivariate data set

    5. linear regression and function fitting:

      - linRegression()    linear regression, y=ax+b, with analytical formula
      - odFit()            fit function with x and y errors (wit package *scipy* *ODR*)
      - xyFit()            fit with with correlated x and y errors,
        profile likelihood and contour lines (module phyFit) 
      - xFit()             fit of parameters to indexed data x_i (module *phyFit*)
      - hFit()             fit of a density to histogram data (module *phyFit*)
      - mFit()             fit of a user-defined cost function, or of a density 
        to unbinned data (module *phyFit*)
      - k2Fit()            fit function with (correlated) errors on x and y  
        with package kafe2
      - k2hFit()            fit of a density to histogram data   
        with package kafe2

    6. simulated data with MC-method:

      - smearData()          add random deviations to input data
      - generateXYdata()     generate simulated data 

    phyFit fitting package for binned and unbinned ML fits and ML fits to (x,y) data

      - mFit()             unbinned ML fit with user-defined negLogL or PDF
      - hFit()             fit to binned histogram data
      - xFit()             fit of parameters to indexed data x_i, with x_i=x_i(x_j, \*par)
      - xyFit()            fit to (x,y) data with y = f(x; \*par) 

    7. helper functions

      - check_function_code()    check Python code before using it in exec() command
      - csv2yaml()               convert csv format to yaml data block
      - plot_xy_from_yaml()      plot (xy) data from yaml file
      - plot_hist_from_yaml()    plot histogram data from yaml file

  """
  # print the above docstring if called
  print(A0_readme.__doc__)
  
# Author:       G. Quast   Dec. 2015
# dependencies: PYTHON v2.7 or >v3.5, numpy, matplotlib.pyplot 
#
# last modified: Jan. May2021
#
#   16-Nov-16    GQ  readPicoscope now also supports .csv export format    
#                GQ  added functions for signal processing/analysis
#   17-Nov-16    GQ  added readCassy for Cassy data in .txt format
#   24-Nov-16    GQ  fixes to FourierSpectrum()
#   07-Dec-16    GQ  changed normalisation of Fourier Amplitudes
#   23-Apr-17    GQ  added auto-correlation function
#                    added readCSV and writeCSV
#   20-May-17    GQ  added readtxt to read data in "txt"-format
#   21-May-17    GQ  simplified read_ColumnData and test_readColumnData
#   22-May-17    GQ  re-implemented readCassy() using readtxt()
#   25-May-17    GQ  allow scalar errors in odFit(), linRegressionXY(),
#                       as in all other fit interfaces
#   27-July-17   GQ  added convolutionEdgefinder() and refactored
#                    convolutionPeakfinder() to use similar components
#   05-Feb-19    CV  added line ending chars to createCSV(), to export
#                    data for example as a latex table
#   07-Feb-19    GQ  merged pull request by CV, vers. 1.0.2
#   27-Jun-19    GQ  fixed readCV and integer arithmetics in Fourier_fft
#   08-Jul-19    GQ  fixed ' ' as deliminter in readtxt()
#   27-Jul-19    GQ  added caption to writeTexTable and f.close to writeCSV()
#   01-Nov-19    GQ  fixed extraction of file extension in readPicoScope
#   05-Nov-20    GQ  changed to GNU GPL, automatic handling of missing uncertainties
#                      in odFit, kFit and k2Fit
#   08-Jan-21    GQ  added fit example with iminuit
#   06-May-21    GQ  new version 1.2.0dev; removed kafe and old iminuit
# ---------------------------------------------------------------------------

import numpy as np, matplotlib.pyplot as plt
from scipy import stats

## ------- section 1: input from text files ------------------------

def readColumnData(fname, cchar='#', delimiter=None, pr=True):
  """read column-data from file

  - input is assumed to be columns of floats
  - characters following <cchar>, and <cchar> itself, are ignored          
  - words with preceding '*' are taken as keywords for meta-data,
    text following the keyword is returned in a dictionary 

  Args:

  * string fnam:      file name
  * int ncols:        number of columns
  * char delimiter:   character separating columns
  * bool pr:          print input to std out if True
  """ 

# -- helper function to filter input lines
  def filter_lines(f, keys, cc='#', delim=None):
    """filter lines for np.loadtxt and 
        extract non-numerical information

    Args:
    * string f:  file name
    * dictionary keys: emtpy dictionary
    * char cc:   comment character

    Modifies:
    * dictionary keys with keywords found in file

    Yields:
    * a valid line with numerical data
    """
    while True:
      line=f.readline()
      if (not line): break # eof, exit
      if cc in line:
        line=line.split(cc)[0] # ignore everything after comment character    
        if (not line): continue # ignore comment lines
      if (not line.isspace()):  # ignore empty lines
         words=line.split()
         if (words[0][0]=="*"): # definition of a key
           keys[words[0]]=' '.join(words[1:]) # get rest of line 
         else:
           # have a valid line, change delimiter to white space
           if (delim is not None): line=line.replace(delim, ' ')
           yield line  # pass line to loadtxt()
#   -- end filter_lines

# -------------------------------------------------------
# define a dictionary for meta data from file
  mdict = {}
  arr = np.loadtxt( filter_lines(open(fname,'r'), 
                    mdict, delim=delimiter, cc=cchar),
                    dtype=np.float32, unpack=True)

# eventually, print out the data we just read:
  if pr:
    print("\n*==* readColumnData: file read successfully")
    print("keywords found:")
    for key in mdict:
      if (mdict[key] is not None): print(key, ':', mdict[key])
    print("data read:")
    for i in range(arr.shape[0]): print(arr[i])

  return arr, mdict

def readCSV(file, nlhead=1, delim=','):
  """read Data in .csv format, skip header lines
  
  Args:
    * file: string, file name 
    * nhead: number of header lines to skip
    * delim: column separator

  Returns:
    * hlines: list of string, header lines
    * data: 2d array, 1st index for columns
  """
# --------------------------------------------------------------------

  # open file for read (if necessary)
  if type(file)==type(' '): f = open(file, 'r') # file is a file name
  else: f=file     # assume input is file handle of an open file 

  hlines=[]
  
  # read header
  for i in range (nlhead):
   hlines.append(f.readline()) # header line(s)

  # read data
  data = np.loadtxt(f, delimiter=delim , unpack=True) # column-wise data
  return hlines, data


def readtxt(file, nlhead=1, delim='\t'):
  """ read floating point data in general txt format
  skip header lines, replace decimal comma, remove special characters
  
  Args:
    * file: string, file name 
    * nhead: number of header lines to skip
    * delim: column separator
  Returns:
    * hlines: list of string, header lines
    * data: 2d array, 1st index for columns
  """
# --------------------------------------------------------------------
# -- helper function to filter input lines
  def specialCharFilter(f, delim):
    """a generator to filter lines read from file
    replace German ',' by '.', remove special characters 

    Args:
      * string f:  file name
    Yields:
      * a valid line with numerical data
    """
    while True:
      l=f.readline()      # read one line
      if (not l): break   # end-of-file reached, exit

    # remove white spaces and control characters, fix German floats 
        # remove leading and trailing white spaces
      l=l.strip()         
        # remove ascii control characters (except delimiter) 
      for i in range(32):
        if delim != chr(i) : l=l.replace(chr(i),'') 
      if l=='': continue        # skip empty lines
        # replace German decimal comma (if not CSV format)
      if delim != ',' :
        filtline=l.replace(',', '.')
      else:
        filtline=l

      yield filtline           # pass filtered line to loadtxt()
#   -- end specialCharFilter

  # open file for read (if necessary)
  if type(file)==type(' '): f = open(file, 'r') # file is a file name
  else: f = file        # assume input is file handle of an open file 

  hlines=[]
  lfilt = specialCharFilter(f, delim) # python generator 
  # read header
  for i in range (nlhead):
    h = next(lfilt)
#    print(h)
    hlines.append(h)  # header line(s)

  # read float data with loadtxt()
  if delim ==' ' or delim =='\t':
    delim = None   # loadtext takes care of white spaces by default
  data = np.loadtxt(lfilt, dtype=np.float32, delimiter=delim, unpack=True)
  return hlines, data


def readPicoScope(file, prlevel=0):
  """
  read Data exported from PicoScope in .txt or .csv format
  
  Args:
    * file: string, file name 
    * prlevel: printout level, 0 means silent

  Returns:
    * units: list of strings, channel units  
    * data: tuple of arrays, channel data

  """
# --------------------------------------------------------------------
#        special treatment to skip/analyze first three lines
  f = open(file, 'r')
  line1=f.readline().strip() # remove leading and trailing white space chars
  line2=f.readline().strip()
  units=line2         # contains the units
  line3=f.readline()  # this is an empty line in PicoScope data

  if file.split('.')[-1]=="csv":
    delim=','
  else:  
    delim='\t'

  units=units.split(delim)
  nc=len(units)
  data = np.loadtxt(f, dtype=np.float32, delimiter=delim, unpack=True)
  if prlevel: 
    print("*==* readPicoScope: %i columns found:"%nc)
    if prlevel>1:
      for i, d in enumerate(data):
        print("     channel %i,  %i values found, unit %s"%(i, len(d), units[i]))
     
  if len(data) != nc:
    print("  !!! number of data columns inconsistent with number of units")
    exit(1)
  else:  
    return units, data

  
def readCassy(file, prlevel=0):
  """read Data exported from Cassy in .txt format
  
  Args:
    * file: string, file name 
    * prlevel: printout level, 0 means silent

  Returns:
    * units: list of strings, channel units  
    * data: tuple of arrays, channel data

  """
# --------------------------------------------------------------------
  delim='\t'                 # Cassy uses <tab> as column delimiter
  hlines, data = readtxt(file, nlhead=5, delim=delim)
  tags = hlines[4].replace('DEF=','').split(delim)
  nc=len(tags)

  if prlevel: 
    print(("*==* readCassy: %i columns found:"%nc))
    if prlevel>1:
      for i, d in enumerate(data):
        print(("     channel %i,  %i values found, tag %s"%(i, len(d), tags[i])))
     
  if len(data) != nc:
    print("  !!! number of data columns inconsistent with number of units")
    exit(1)
 
  return tags, data


def labxParser(file, prlevel=1):
  """read files in xml-format produced with Leybold CASSY
   
  Args:
     * file:  input data in .labx format
     * prlevel: control printout level, 0=no printout
 
  Returns:
     * list of strings: tags of measurement vectors
     * 2d list:         measurement vectors read from file 
  """
# --------------------------------------------------------------------
# dependencies: xml.etree.ElementTree
#
#  30-Oct-16  initial version
# changes :
# --------------------------------------------------------------------
  import xml.etree.ElementTree as ET
  import numpy as np, matplotlib.pyplot as plt
  import sys

  root = ET.parse(file).getroot()
  if root.tag != 'cassylab':
    print(" !!! only cassylab supported - exiting (1) !")
    sys.exit(1)    
  else:
    if(prlevel): print("\n\n*==* labxParser: name of XML root object:\n",\
    ' ', root.tag, root.attrib)
#
# some sanity checks wether we got the right kind of input
  if not root.findall('cassys'):
    print(" !!! no tag 'casssys' found  - exiting (2) !")
    sys.exit(2)    
  if not root.findall('ios'):
    print("! !!! tag 'ios' not found exiting (3) !")
    sys.exit(3)    
#
# print header info if requested 
  if (prlevel>1):
    childvals=[]
    childtags=[]
    for child in root:
      childtags.append(child.tag)
      childvals.append(child.attrib)
    print("    %i children found, " %(len(childtags)), end=' ') 
    print("tags and attributes: \n", end=' ')
    for i in range(len(childtags)):
      print('   ', childtags[i],' : ', childvals[i])

  if(prlevel>2):
    print('\n *==*  Details:') 
    print(" ** found tag 'ios', configuration:")
    for ios in root.findall('ios'):
      print('   ', ios.tag, ios.attrib)
    print("   measurement settings:")
    i=0
    for io in ios.findall('io'): 
      i+=1      
      print("  --> io %i:"%i, io.attrib)
      for qs in io.iter('quantities'): print('   ', qs.tag, qs.attrib)
      for q in qs.iter('quantity'): print('   ', q.tag, q.attrib)

  if(prlevel>2):
    if root.findall('calcs'):
      print("\n ** found tag 'calcs', calc settings:")
      for calcs in root.findall('calcs'):
        i=0
        for calc in calcs.findall('calc'): 
          i+=1      
          print("  --> calc %i:"%i, calc.attrib)

# ---- collect data in vectors 
  # cassylab stores data under the tag "channels:channel:values", 
  #    search for and extract data from xml structure
  varray=[]
  vnames=[]
  vsymbols=[]
  vunits=[]
  vtags=[]
  iv=0
  ic=0
  for clist in root.iter('channels'):
    for c in clist:
      ic+=1
      quantity=c.find('quantity').text
      vnames.append(quantity)
      symbol=c.find('symbol').text
      if symbol is None: symbol=''
      vsymbols.append(symbol)
      unit=c.find('unit').text
      if unit is None: unit=''
      vunits.append(unit)
      vtag = '%i:'%ic + quantity + ':' + symbol + ':' + unit
      vtags.append(vtag)
      if(prlevel>1): 
        print("   --> new channel found", vtag) 
        if(prlevel>2): print("     ", c.attrib)

      values=c.find('values')
      if(prlevel>2): print("     number of values: ", values.attrib)
      varray.append([])
      for v in values:
        varray[iv].append(np.float32(v.text))
      iv+=1

  if (prlevel): 
    print("*==* labxParser:  %i value lists found"%iv)
    for tag in vtags:
      print("  ", tag)
    print("\n\n")

  return vtags, varray


def writeCSV(file, ldata, hlines=[], fmt='%.10g',
              delim=',', nline='\n', **kwargs):
  """write data in .csv format, including header lines
  
  Args:
    * file: string, file name 
    * ldata: list of columns to be written
    * hlines: list with header lines (optional)
    * fmt: format string (optional)
    * delim: delimiter to separate values (default comma)
    * nline: newline string

  Returns: 
    * 0/1  for success/fail
  """

# --------------------------------------------------------------------

  # open file for read (if necessary)
  if type(file)==type(' '): f = open(file, 'w') # file is a file name
  else: f=file     # assume input is file handle of an open file 
  
  #check if \n is contained in newline, if not add it
  if "\n" not in nline:
    nline += "\n"

  if type(hlines)==type(' '):
     f.write(hlines+nline)
  elif type(hlines)==type([]):
    for i in range(len(hlines)):
      f.write(hlines[i]+nline)

  try:
    np.savetxt(f, np.array(ldata).transpose(),
                fmt=fmt, delimiter=delim, newline=nline, **kwargs)
    f.close()
    return 0
  except:
    return 1

def writeTexTable(file, ldata, cnames=[], caption='', fmt='%.10g'):
  """write data formatted as latex tabular

  Args:
    * file: string, file name
    * ldata: list of columns to be written
    * cnames: list of column names (optional)
    * caption: LaTeX table caption (optional)
    * fmt: format string (optional)

  Returns:
    * 0/1 for success/fail
  """

  delim = " & "
  nline = "\\\\\n"

  #create header for latex tabular environment
  head = "\\begin{tabular}{"+len(ldata)*"c"+"}\n\\hline\n"
  if type(cnames)==type(''):
    head += cnames 
  elif type(cnames)==type([]) and len(cnames)>0:
    for element in cnames:
      head += element + " & "
    head=head[:-3] +'\\\\'  # remove last '&' and replace by '\\'
    head+='\n\\hline\n%' # last header line is a comment line
  #create footer
  foot = '\n\\hline\n\\end{tabular}'
  if caption != '': foot += '\n\\caption{' + caption +'}'
  foot+='\n%' # last footer line is a comment line
  # write to file
  return writeCSV(file, ldata, fmt=fmt, delim=delim, nline=nline,
                  header=head, footer=foot, comments='')


def round_to_error(val, err, nsd_e=2):
  """round float *val* to corresponding number of significant digits  
  as uncertainty *err*

  Arguments:
    * val, float: value
    * err, float: uncertainty of value
    * nsd_e, int: number of significant digits of err
 
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

def ustring(v, e, pe=2):
  """v +/- e as formatted string 
  with number of significant digits corresponding to 
  precision pe of uncertainty 

  Args:
    * v: value
    * e: uncertainty
    * pe: precision (=number of significant digits) of uncertainty

  Returns:
    * string: <v> +/- <e> with appropriate number of digits
  """

  # format string for printout (releys on #.g format)
  fmttxt="{:#.{pv}g}+/-{:#.{pe}g}"
  # get number of significant digits of v and rounded values 
  nd, _v, _s = round_to_error(v, e, pe)
  return fmttxt.format(_v, _s, pv=nd, pe=pe)


## ------- section 2: statistics  -----------------------

def wmean(x, sx, V=None, pr=True):
  """ weighted mean of np-array x with uncertainties sx 
  or covariance matrix V; if both are given, sx**2 is added 
  to the diagonal elements of the covariance matrix
 
  Args:
    * x: np-array of values
    * sx: np-array uncertainties
    * V: optional, covariance matrix of x
    * pr: if True, print result

  Returns:
    * float: mean, sigma 
  """
  
  if V is None:
    w = 1/sx**2
  else:
    cov = V 
    if sx is not None:  # add independent errors to diagonal of V
      np.fill_diagonal(cov, V.diagonal() + sx*sx) 
    # calculate inverse of covariance matrix
    covI = np.mat(cov).I
    w = np.sum(covI, axis=0)
#
  sumw = np.sum(w)
  mean = np.inner(w,x)/sumw
  smean = np.sqrt(1./sumw)
  # eventually, print out the data we just read:
  if pr:
    print("\n weighted mean = %.3g +/- %.3g"%(mean, smean))
  return mean, smean

def BuildCovarianceMatrix(sig, sigc=[]):
  """Construct a covariance matrix 
  from independent and correlated error components

  Args: 
   * sig: iterable of independent errors 
   * sigc: list of iterables of correlated uncertainties
  
  Returns: 
   covariance Matrix as numpy-array
  """
  
  # construct a numpy array with diagonal elements 
  su = np.array(sig)
  V = np.diagflat(su*su)

  # if given, add off-diagonal components
  if sigc != []:
    for s in sigc:
      sc = np.array(s)
      V += np.outer(sc, sc)
  return V

def Cov2Cor(V):
  """Convert a covariance matrix into diagonal errors + correlation matrix

  Args: 
   * V: covariance matrix as numpy array
  
  Returns: 
   * diag uncertainties (sqrt of diagonal elements)
   * C: correlation matrix as numpy array
  """
  diag = np.sqrt(np.diag(V))  
  C = V/np.outer(diag, diag)             
  return diag, C


def Cor2Cov(sig, C):
  """Convert a correlation matrix and error into covariance matrix

  Args: 
   * sig: 1d numpy array of correlated uncertainties
   * C: correlation matrix as numpy array
  
  Returns: 
   * V: covariance matrix as numpy array
  """
  V = C * np.outer(sig, sig)             
  return V

# -------------------------------------------------------

def chi2prob(chi2, ndf):
  """ chi2-probability
 
  Args:
    * chi2: chi2 value
    * ndf: number of degrees of freedom

  Returns:
    * float: chi2 probability
  """

  return 1.- stats.chi2.cdf(chi2, ndf)

# -------------------------------------------------------

def propagatedError(function, pvals, pcov):
  """ determine propageted uncertainty (with covariance matrix)
 
  Formula: 
  Delta = sqrt( sum_i,j (df/dp_i df/dp_j Vp_i,j) )

  Args:
    * function: function of parameters pvals, 
      a 1-d array is also allowed, eg. function(\*p) = f(x, \*p)
    * pvals: parameter values
    * pcov: covariance matrix (or uncertainties) of parameters

  Returns:
    * uncertainty Delta( function(\*par) )
  """

  # check if pcov is a matrix; if not, uncertainties and construct matrix
  cov = np.asarray(pcov)
  if len(np.shape(cov)) == 1 :
    cov = np.diag(cov * cov)
    
  # first, calculate partial derivatives of model w.r.t parameters    
  #   set fractional step size 1% of parameter uncertainty
  stepsize = 0.01  
  dp = stepsize * np.sqrt(np.diagonal(cov))
  #   derivative df/dp_j
  dfdp = []
  p_plus = np.array(pvals, copy=True)
  p_minus = np.array(pvals, copy=True)
  for j in range(len(pvals)): 
    p_plus[j] = pvals[j] + dp[j]
    p_minus[j] = pvals[j] - dp[j]
    dfdp.append( 0.5 / dp[j] * (
                  function(*p_plus) - function(*p_minus) )
                 )
    p_plus[j] = pvals[j] 
    p_minus[j] = pvals[j]
  dfdp = np.array(dfdp)
  
  # then, calculate square of uncertainty on function value
  if len(np.shape(dfdp)) == 1:  # model returned a scalar
    Delta2 = np.sum(np.outer(dfdp, dfdp) * cov)
  elif len(np.shape(dfdp)) == 2:   # function returns array
    dim = len(dfdp[0])
    Delta2 = np.empty(dim)
    for i in range(dim):
      Delta2[i] = np.sum(np.outer(dfdp[:,i], dfdp[:,i]) * cov)
  else:
    print("!!! PhyPraKit.propagatedErrors(): cannot handle >1d arrays")
    return None
  return np.sqrt(Delta2) 

def getModelError(x, model, pvals, pcov):
  """ determine uncertainty of model at x from parameter uncertainties  
    
  Formula: 
    Delta(x) = sqrt( sum_i,j (df/dp_i(x) df/dp_j(x) Vp_i,j) )

  Args:
    * x: scalar or 1d-array of x values
    * model: model function
    * pvals: parameter values
    * covp: covariance matrix of parameters

    Returns:
    * model uncertainty/ies, same length as x
  """
  
  def fun(*pvals):
    return model(x, *pvals)

  return propagatedError(fun, pvals, pcov)

## ------- section 3: signal processing -----------------

def offsetFilter(a):
  ''' 
  correct an offset in array a 
  (assuming a symmetric signal around zero)
  by subtracting the mean
  '''
  return a-a.mean()

def meanFilter(a, width=5):
  ''' 
  apply a sliding average to smoothen data, 

  method:
    value at index i and int(width/2) neighbours are averaged
    to from the new value at index i

    Args:
      * a: np-array of values
      * width: int, number of points to average over
        (if width is an even number, width+1 is used)
 
    Returns:
      * av  smoothed signal curve
  '''
# -----------------------------------------------
  l=len(a)
  av = np.zeros(l) 
  k=int(width/2)
  for i in range(k, l-k+1):
    av[i]= sum(a[i-k:i+k+1])/(2*k+1)

  return av

def resample(a, t=None, n=11):
  ''' 
  perform average over n data points of array a, 
  return reduced array, eventually with corresponding time values 

  method:
    value at index `i` and `int(width/2)` neighbours are averaged
    to from the new value at index `i`

    Args:
      * a, t: np-arrays of values of same length
      * width: int, number of values of array `a` to average over
        (if width is an even number, width+1 is used)
 
    Returns:
      * av: array with reduced number of samples
      * tav:  a second, related array with reduced number of samples 
  '''
  k = int(n/2)
  nav = int( len(a) /(2*k+1))
  av = np.zeros(nav) 
  if t is not None: 
    tav=np.zeros(nav)
  j=0
  for i in range(k, len(a)-k, 2*k+1):
    av[j]= sum(a[i-k:i+k+1])/(2*k+1)
    if t is not None: tav[j]=t[i]
    j+=1
  
  if t is not None:
    return av, tav
  else: 
    return av

def Fourier_fft(t, a):
  ''' 
  Fourier transform of the amplitude spectrum a(t) 
  
  method: 
    uses `numpy.fft` and `numpy.fftfreq`; 
    output amplitude is normalised to number of samples; 

    Args:
      * t: np-array of time values
      * a: np-array amplidude a(t)
 
    Returns:
      * arrays f, a_f: frequencies and amplitudes
  '''
# -----------------------------------------------
  from numpy.fft import fft, fftfreq

  n = len(t)
  dt = (t[-1]-t[0])/(n-1.)       # time step
  freq = fftfreq(n, dt)[:n//2]   # only positive frequencies
  amp = abs(fft(a))[:n//2]*2./n

  return freq, amp


def FourierSpectrum(t, a, fmax=None):
  '''
  Fourier transform of amplitude spectrum a(t), for equidistant sampling
  times (a simple implementation for didactical purpose only,
  consider using ``Fourier_fft()`` )

    Args:
      * t: np-array of time values
      * a: np-array amplitude a(t)
 
    Returns:
      * arrays freq, amp: frequencies and amplitudes
  '''
# -----------------------------------------------

  n = len(t)      
  T = t[-1 ]-t[0] # total time covered by sample
  df = 1./T       # smallest frequency and frequency step
  dt = T / (n-1.) # time step, 1/2 1/dt is largest frequency
  fmx = 0.5/dt    # Nyquist Theorem: n/2 frequency components
  if fmax is not None:
    if fmax>fmx:
       print("!!! FourierSpectrum: fmax too large, set to ", fmx)
    fmx = min(fmax, fmx)

  freq = np.arange(df, fmx, df)

  # calculate coefficients
  amp = np.zeros(len(freq))
  i=0
  for f in freq:
    omega = 2. * np.pi * f
    s=sum(a * np.sin(omega * t))*2./n
    c=sum(a * np.cos(omega * t))*2./n
    amp[i] = np.sqrt(s**2 + c**2)
    i+=1
  # alternative implementation without explicitly writing the loop
  #omegat=np.outer(2.*np.pi*freq, t)  # this uses much memory !!
  #s = np.matmul(np.sin(omegat), a)*2./n
  #c = np.matmul(np.cos(omegat), a)*2./n
  #amp = np.sqrt(s**2 + c**2)

  return freq, np.array(amp)


def simplePeakfinder(x, a, th=0.):
  ''' 
  find positions of all maxima (peaks) in data
    x-coordinates are determined from weighted average over 3 data points

  this only works for very smooth data with well defined extrema
    use ``convolutionPeakfinder`` or ``scipy.signal.argrelmax()`` instead

    Args:
      * x: np-array of positions
      * a: np-array of values at positions x
      * th: float, threshold for peaks
 
    Returns:
      * np-array: x positions of peaks as weighted mean over neighbours
      * np-array: y values corresponding to peaks 
  '''
# -----------------------------------------------
# work on normalized input
  y = (a-min(a))/(max(a)-min(a))
# 
  xpeak=[]
  apeak=[]
  if y[0]-y[1]>th and y[0]-y[2]>th:
    xpeak.append(x[0])    
    apeak.append(a[0])    
  for i in range(1,len(x)-1):
    if y[i]-y[i-1]>th and y[i]-y[i+1]>th: 
      xpeak.append(sum(x[i-1:i+1]*y[i-1:i+1])/sum(y[i-1:i+1]))    
      apeak.append(a[i])    
  if y[-1]-y[-2]>th and y[-1]-y[-3]>th:
    xpeak.append(x[-1])    
    apeak.append(a[-1])    

  return np.array(xpeak), np.array(apeak)

def convolutionFilter(a, v, th=0.):
  ''' 
  convolute normalized array with template function and return maxima

  method: 
    convolute array a with a template and return extrema of 
    convoluted signal, i.e. places where template matches best

  Args:
    * a: array-like, input data
    * a: array-like, template 
    * th: float, 0.<= th <=1., relative threshold for places of
      best match above (global) minimum

  Returns:
    * pidx: list, indices (in original array) of best matches

  '''

  anrm = (a-min(a))/(max(a)-min(a))
  c = np.correlate( anrm, v, mode='same')
  #c = np.convolve( anrm, np.flipud(v), 'same')
  # remark: need reversed ordering of v to use np.convolve() for this purpose

# store places of best agreement with the template 
  pidx=[]
  for i in range (1, len(anrm)-1):
    if c[i]>0. and c[i]-c[i-1]>=0. and c[i]-c[i+1]>0. and anrm[i] > th : 
      pidx.append(i) 
  return pidx

def convolutionPeakfinder(a, width=10, th=0.0):
  ''' 
  find positions of all Peaks in data 
    (simple version for didactical purpose, 
    consider using ``scipy.signal.find_peaks_cwt()`` )

  method: 
    convolute array a with rectangular template of given width and
    return extrema of convoluted signal, i.e. places where 
    template matches best

  Args:
    * a: array-like, input data
    * width: int, width of signal to search for
    * th: float, 0.<= th <=1., relative threshold for peaks above (global)minimum

  Returns:
    * pidx: list, indices (in original array) of peaks
  ''' 

#construct a (rectangular) template for a peak
  k=int(width/2)
  v = np.array(\
        [-0.5 for i in range(k)] +\
        [0.5 for i in range(2*k+1)] +\
        [-0.5 for i in range(k)], 
               dtype=np.float32 )
  return convolutionFilter(a, v, th=th)

def convolutionEdgefinder(a, width=10, th = 0.):
  ''' 
  find positions of maximal positive slope in data 

  method: 
    convolute array `a` with an edge template of given width and
    return extrema of convoluted signal, i.e. places of rising edges

  Args:
    * a: array-like, input data
    * width: int, width of signal to search for
    * th: float, 0.<= th <=1., relative threshold above (global)minimum

  Returns:
    * pidx: list, indices (in original array) of rising edges
  ''' 

#construct a (rectangular) template for an edge
  k=int(width/2)
  v = np.array(\
        [-0.5 for i in range(k)] +\
               [0.] +\
        [0.5 for i in range(k)], 
               dtype=np.float32 )
  return convolutionFilter(a, v, th=th)

def autocorrelate(a):
  '''calculate auto-correlation function of input array 

     method: for array of length l, calculate 
     a[0]=sum_(i=0)^(l-1) a[i]*[i] and 
     a[i]= 1/a[0] * sum_(k=0)^(l-i) a[i] * a[i+k-1] for i=1,l-1 

     Args:
       * a: np-array 

     Returns 
       * np-array of len(a), the auto-correlation function
  '''

  l=len(a)
  rho=np.zeros(l)
  for i in range(1, l):
    rho[i] = np.inner(a[i:], a[:-i])
  rho[0] = np.inner(a, a)
  rho = rho/rho[0]

  return rho


## ------- section 4: histograms in 1d and 2d ----------------------

def barstat(bincont, bincent, pr=True):
  """statistics from a bar chart (histogram) 
     with given bin contents and bin centres

     Args:
       * bincont: array with bin content
       * bincent: array with bin centres

     Returns:
       * float: mean, sigma and sigma on mean    
  """
  mean=sum(bincont*bincent)/sum(bincont)
  rms=np.sqrt(sum(bincont*bincent**2)/sum(bincont) - mean**2)
  smean=rms/np.sqrt(sum(bincont))
  if pr: 
    print('bar chart statistics:\n'\
'   mean=%g, sigma=%g, sigma mean=%g\n' %(mean,rms,smean))
  return mean, rms, smean

def nhist(data, bins=50, xlabel='x', ylabel='frequency') :
# ### own implementation of one-dimensional histogram (numpy + matplotlib) ###
  """ Histogram.hist
      show a one-dimensional histogram 

      Args:
        * data: array containing float values to be histogrammed
        * bins: number of bins
        * xlabel: label for x-axis
        * ylabel: label for y axix

      Returns:
        * float arrays: bin contents and bin edges
  """

  bc,be = np.histogram(data,bins) # histogram data
  bincent=(be[:-1] + be[1:])/2.
  w=0.9*(be[1]-be[0])
  plt.bar(bincent,bc,align='center',width=w,facecolor='b',alpha=0.75) #
  plt.xlabel(xlabel,size='x-large') # ... for x ...
  plt.ylabel(ylabel,size='x-large') # ... and y axes
#  plt.show()
  return bc, be

def histstat(binc, bine, pr=True):
  """ Histogram.histstat
    calculate mean, standard deviation and uncertainty on mean 
    of a histogram with bin-contents `binc` and bin-edges `bine`
 
    Args:
      * binc: array with bin content
      * bine: array with bin edges

    Returns:
      * float: mean, sigma and sigma on mean    
  """

  bincent =(bine[1:]+bine[:-1])/2 # determine bincenters
  mean=sum(binc*bincent)/sum(binc)
  rms=np.sqrt(sum(binc*bincent**2)/sum(binc) - mean**2)
  sigma_m=rms/np.sqrt(sum(binc))
  if pr: 
    print('hist statistics:\n'\
'   mean=%g, sigma=%g  sigma of mean=%g\n' %(mean,rms,sigma_m))
  return mean, rms, sigma_m

def nhist2d(x, y, bins=10, xlabel='x axis', ylabel='y axis', clabel='counts'):
# ### own implementation of two-dimensional histogram ###
  """ Histrogram.hist2d
    create and plot a 2-dimensional histogram  

    Args:
      * x: array containing x values to be histogrammed
      * y: array containing y values to be histogrammed
      * bins: number of bins
      * xlabel: label for x-axis
      * ylabel: label for y axix
      * clabel: label for colour index

    Returns:
      * float array: array with counts per bin
      * float array: histogram edges in x
      * float array: histogram edges in y
  """
  H2d, xed, yed = np.histogram2d(x,y,bins) # numpy 2d histogram function
  Hpl = np.rot90(H2d)  # rotate, ...
  Hpl = np.flipud(Hpl) # ... flip, ... 
  Hpl = np.ma.masked_where(Hpl==0,Hpl) # ... and mask zero values, ...
  im = plt.pcolormesh(xed,yed,Hpl,cmap='Blues') # ... then make plot
  cbar = plt.colorbar() # show legend 
  cbar.ax.set_ylabel(clabel) # print labels for legend, ...
  plt.xlabel(xlabel) # ... for x ...
  plt.ylabel(ylabel) # ... and y axes
#  plt.show()
  return H2d, xed, yed

def hist2dstat(H2d, xed, yed, pr=True):
  """
    calculate statistical information from 2d Histogram

    Args:
      * H2d: histogram array (as returned by histogram2d)
      * xed: bin edges in x
      * yed: bin edges in y

    Returns:
      * float: mean x
      * float: mean y 
      * float: variance x
      * float: variance y
      * float: covariance of x and y
      * float: correlation of x and y
  """
  bcx=(xed[:-1]+xed[1:])/2 
  bcy=(yed[:-1]+yed[1:])/2
  sumxy,sumx,sumx2,sumy,sumy2,sum=0.,0.,0.,0.,0.,0.
  for ix in range(0,len(bcx)):
    for iy in range(0,len(bcy)):
      sumxy += H2d[ix,iy]*bcx[ix]*bcy[iy]
      sumx += H2d[ix,iy]*bcx[ix]
      sumx2 += H2d[ix,iy]*bcx[ix]*bcx[ix]
      sumy += H2d[ix,iy]*bcy[iy]
      sumy2 += H2d[ix,iy]*bcy[iy]*bcy[iy]
      sum += H2d[ix,iy]
  meanx = sumx/sum
  varx = (sumx2/sum-meanx*meanx)
  meany = sumy/sum
  vary = (sumy2/sum-meany*meany)
  cov = (sumxy/sum-meanx*meany) 
  cor = cov/np.sqrt(varx*vary)
  if pr: 
    print('hist2d statistics:\n'\
'   <x>=%g, <y>=%g\n\
    var_x=%.2g, var_y=%.2g\n\
    cov=%.2g, cor=%.2g\n'\
    %(meanx,meany,varx,vary,cov,cor))
  return meanx,meany,varx,vary,cov,cor

def profile2d(H2d, xed, yed):
  """
    generate a profile plot from 2d histogram:
      - mean y at a centre of x-bins, standard deviations as error bars

    Args:
      * H2d: histogram array (as returned by histogram2d)
      * xed: bin edges in x
      * yed: bin edges in y

    Returns:
      * float: array of bin centres in x
      * float: array mean
      * float: array rms
      * float: array sigma on mean
  """
  mean_y=[]
  rms_y=[]
  sigm_y=[]
  for ix in range(len(xed)-1):
    m, rms, sigm =histstat(H2d[ix,:], yed, False) 
    mean_y.append(m)
    rms_y.append(rms)
    sigm_y.append(sigm)
    bcx=(xed[:-1] + xed[1:])/2.
  plt.errorbar(bcx, mean_y, xerr=0., yerr=rms_y, fmt='bo')
  plt.errorbar(bcx, mean_y, xerr=0., yerr=sigm_y, fmt='k,',linewidth=2)
  return bcx, mean_y, rms_y, sigm_y
 
def chi2p_indep2d(H2d, bcx, bcy, pr=True):
  """
    perform a chi2-test on independence of x and y

    method: chi2-test on compatibility of 2d-distribution, f(x,y),
    with product of marginal distributions, f_x(x) * f_y(y)

    Args:
      * H2d: histogram array (as returned by histogram2d)
      * bcx: bin contents x (marginal distribution x)
      * bcy: bin contents y (marginal distribution y)

    Returns:
      * float: p-value w.r.t. assumption of independence
  """
  threshold=5. # minimum number of expected entries in a bin 
  Ntot=sum(bcx)
  chi2=0.
  ndf=0
  usedx=np.zeros(len(bcx))
  usedy=np.zeros(len(bcy))
  for ix in range(len(bcx)):
    for iy in range(len(bcy)):
      Nexp=bcx[ix]*bcy[iy]/Ntot
      if Nexp>threshold:
        usedx[ix]=1.
        usedy[iy]=1.
        ndf+=1.
        chi2+=((H2d[ix,iy]-Nexp)**2)/Nexp
  ndf-=(sum(usedx) + sum(usedy))
  # print 'chi2, ndf, npar =',chi2, ndf
  pval=1.- stats.chi2.cdf(chi2, ndf)
  if pr: 
    print('p-value of chi2-independence test: %.2g%%\n'%(pval*100.))
  return pval


def plotCorrelations(vals, names=None):  
  """plot histograms and sctter plots of value pairs as array of axes

  Args:

  - vals: list of arrays [[v1_1, ...], ..., [vn_1, ...]] of float, input data
  - names: list of labels for variables v1 to vn

  Returns:

  - figure
  - axarray: array of axes
  """
  npar=len(vals)
  nd = len(vals[0])
  if names is None:
    names = ['var{:d}'.format(i) for i in range(npar)]

  fig, axarr = plt.subplots(npar, npar, figsize=(3. * npar, 3.1 * npar))
  fig.tight_layout()
  fig.subplots_adjust(top=0.92, bottom=0.1, left=0.1, right=0.95,
                      wspace=0.33, hspace=0.3)  
  nb1= int(min(50, nd/10))
  nb2= int(min(50, nd/10))
    
  ip = -1
  for i in range(0, npar):
    ip += 1
    jp = -1
    for j in range(0, npar):
      jp += 1
      if ip > jp:
        axarr[jp, ip].axis('off')      # set empty space
      elif ip == jp:
        ax=axarr[jp, ip]
        bc, be, _ = ax.hist(vals[ip], nb1) # plot 1d-histogram
        ax.set_xlabel(names[ip])
        ax.locator_params(nbins=5) # number of axis labels
        m, s, sm = histstat(bc, be, False)  # calculate statistics
        ax.axvline(m, color='orange', linestyle='--')
        nsd, _m, _sm = round_to_error(m, sm)
        ax.text(0.75, 0.85,                
                '$\\mu=${:#.{p}g}\n'.format(_m, p=nsd) +
                '$\\sigma=${:#.2g}\n'.format(s) +
                '$\\sigma_\\mu=${:#.2g}'.format(sm),
                transform=ax.transAxes,
                backgroundcolor='white')
      else:
        # 2d-plot to visualise correlations
        ax=axarr[jp, ip]
        H, xe, ye, _ = ax.hist2d(vals[jp], vals[ip], nb2, cmap='Blues')
        ax.set_xlabel(names[jp])
        ax.set_ylabel(names[ip], labelpad=-2)
        ax.locator_params(nbins=5) # number of axis labels
        mx, my, vx, vy, cov, cor = hist2dstat(H,xe,ye,False)
        ax.text(0.33, 0.85, '$\\rho$=%.2f' %cor,
                  transform=ax.transAxes,
                  backgroundcolor='white')
  # set 1st empty space as current axis
  ax=axarr[0, 1]
  plt.sca(ax) 
  return fig, axarr

## ------- section 5: (linear) regression and fits ------------------

def linRegression(x, y, sy=None):
  """linear regression y(x) = ax + b 

  method: 
    analytical formula

  Args:
  * x: np-array, independent data
  * y: np-array, dependent data
  * sy: scalar or np-array, uncertainty on y

  Returns:
  * float: a     slope
  * float: b     constant
  * float: sa  sigma on slope
  * float: sb  sigma on constant
  * float: cor   correlation
  * float: chi2  \chi-square
  """

  # set y-errors to 1. if not given
  if sy is None:
    sy=np.ones(len(y))
    print('\n!**! No y-errors given -> parameter errors from fit are meaningless!\n')

  # calculate auxilary quantities
  S1  = sum(1./sy**2)
  Sx  = sum(x/sy**2)
  Sy  = sum(y/sy**2)
  Sxx = sum(x**2/sy**2)
  Sxy = sum(x*y/sy**2)
  D = S1*Sxx-Sx**2

  # calculate results:
  a   = (S1*Sxy-Sx*Sy)/D # slope
  b   = (Sxx*Sy-Sx*Sxy)/D # constant
  sa  = np.sqrt(S1/D)
  sb  = np.sqrt(Sxx/D)
  cov   = -Sx/D
  cor  = cov/(sa*sb)
  chi2  = sum(((y-(a*x+b))/sy)**2)

  return a, b, sa, sb, cor, chi2


def odFit(fitf, x, y, sx=None, sy=None, p0=None):
  """fit an arbitrary function with errors on x and y
  uses numerical "orthogonal distance regression" from package scipy.odr

  Args:
  * fitf: function to fit, arguments (array:P, float:x)
  * x:  np-array, independent data
  * y:  np-array, dependent data
  * sx: scalar or np-array, uncertainty(ies) on x      
  * sy: scalar or np-array, uncertainty(ies) on y
  * p0: array-like, initial guess of parameters

  Returns:
  * np-array of float: parameter values
  * np-array of float: parameter errors
  * np-array: cor   correlation matrix 
  * float: chi2  \chi-square
  """  
  from scipy.optimize import curve_fit
  from scipy import odr

  # define wrapper for fit function in ODR format
  def fitf_ODR(p, x):
    return fitf(x, *p)

  # set y-errors to 1. if not given
  if sy is None:
    sy=np.ones(len(y))
    print('\n!**! No y-errors given -> parameter errors from fit are meaningless!\n')
  # transform uncertainties to numpy-arrays, if necessary
  if sx is not None:
    if not hasattr(sx,'__iter__'): sx=sx*np.ones(len(x))
  if not hasattr(sy,'__iter__'): sy=sy*np.ones(len(y))

  # perform a simple fit with y-errors only to obtatain start values 
  par0, cov0 = curve_fit( fitf, x, y, sigma=sy, absolute_sigma=True, p0=p0 )
  #print '*==* result from curve fit:'
  #print ' -> par= ', par0
  #print ' -> pare= ', np.sqrt(np.diag(cov0))

  if(sx is None or not np.sum(sx)):          # if no x-errors, we are done
    pare=np.sqrt(np.diag(cov0))
    cor = cov0/np.outer(pare,pare)
    chi2 = np.sum(((fitf(np.array(x), *par0) - y)/sy)**2)
    return par0, pare, cor, chi2
  else:  # use ODR package
    mod = odr.Model(fitf_ODR)
    dat = odr.RealData(x, y, sx, sy)
    odrfit = odr.ODR(dat, mod, beta0 = par0)
    r = odr.ODR.run(odrfit)
    par=r.beta
    cov=r.cov_beta
    pare=np.sqrt(np.diag(cov))
    cor = cov/np.outer(pare, pare) 
    ndf = len(x)-len(par)
    chi2 = r.res_var*ndf

    return par, pare, cor, chi2


def k2Fit(func, x, y,
      sx=None, sy=None, srelx=None, srely=None, 
      xabscor=None, yabscor=None, xrelcor=None, yrelcor=None,
      ref_to_model=True, constraints=None, p0=None, dp0=None, limits=None,
      plot=True, axis_labels=['x-data', 'y-data'], data_legend = 'data',
      model_expression=None, model_name=None, 
      model_legend = 'model', model_band = r'$\pm 1 \sigma$',           
      fit_info=True, plot_band=True, asym_parerrs=True, plot_cor=False,
      showplots=True, quiet=True):

  """Fit an arbitrary function func(x, \*par) to data points (x, y) 
  with independent and correlated absolute and/or relative errors 
  on x- and y- values with package iminuit.

  Correlated absolute and/or relative uncertainties of input data 
  are specified as numpy-arrays of floats; they enter in the 
  diagonal and off-diagonal elements of the covariance matrix. 
  Values of 0. may be specified for data points not affected
  by a correlated uncertainty. E.g. the array [0., 0., 0.5., 0.5]
  results in a correlated uncertainty of 0.5 of the 3rd and 4th 
  data points. Providing lists of such array permits the construction
  of arbitrary covariance matrices from independent and correlated
  uncertainties uncertainties of (groups of) data points.

  Args:
    * func: function to fit
    * x:  np-array, independent data
    * y:  np-array, dependent data

    components of uncertainty (optional, use None if not relevant)

    single float, array of length of x, or a covariance matrix
      * sx: scalar, 1d or 2d np-array, uncertainty(ies) on x      
      * sy: scalar, 1d or 2d np-array, uncertainty(ies) on y

    single float or array of length of x
      * srelx: scalar or 1d np-array, relative uncertainties x
      * srely: scalar or 1d np-array, relative uncertainties y

    single float or array of length of x, or a list of such objects, 
      used to construct a covariance matrix from components

      * xabscor: scalar or 1d np-array, absolute, correlated error(s) on x
      * yabscor: scalar or 1d np-array, absolute, correlated error(s) on y
      * xrelcor: scalar or 1d np-array, relative, correlated error(s) on x
      * yrelcor: scalor or 1d np-array, relative, correlated error(s) on y

    fit options
      * ref_to_model, bool: refer relative errors to model if true,
        else use measured data
      * p0: array-like, initial guess of parameters
      * dp0: array-like, initial guess of parameter uncertainties
      * parameter constraints: (name, value, uncertainty)
      * limits: (nested) list(s) (name, min, max) 

    output options
      * plot: flag to switch off graphical output
      * axis_labels: list of strings, axis labels x and y
      * data_legend: legend entry for data points
      * model_name: latex name for model function
      * model_expression: latex expression for model function
      * model_legend: legend entry for model
      * model_band: legend entry for model uncertainty band
      * fit_info: controls display of fit results on figure
      * plot_band: suppress model uncertainty-band if False
      * asym_parerrs: show (asymmetric) errors from profile-likelihood scan
      * plot_cor: show profile curves and contour lines
      * showplots: show plots on screen, default = True
      * quiet: controls text output

  Returns:
    * list: parameter names
    * np-array of float: parameter values
    * np-array of float: negative and positive parameter errors
    * np-array: cor   correlation matrix 
    * float: chi2  \chi-square

  """  

  # for fit with kafe2
  from kafe2 import XYContainer, Fit, Plot, ContoursProfiler

  # create a data container
  dat = XYContainer(x, y)
  # - provide text for labeling ...      
  dat.label = data_legend
  dat.axis_labels = axis_labels

  # - add all error sources  
  if sy is None and srely is None:
    sy=np.ones(len(y))
    print('\n!**! No y-errors given, all assumed to be 1.0\n',
          '-> consider scaling of parameter errors with sqrt(chi^2/Ndf)\n')

  sy=np.asarray(sy)
  if sy.ndim == 2:
    dat.add_matrix_error(axis='y', err_matrix=sy, matrix_type='covariance')
  else:
    dat.add_error(axis='y', err_val=sy)
  
  if sx is not None:
    sx=np.asarray(sx)
    if sx.ndim == 2:
      dat.add_matrix_error(axis='x', err_matrix=sx, matrix_type='covariance')
    else:
      dat.add_error(axis='x', err_val=sx)

  if srelx is not None: 
    dat.add_error(axis='x', err_val=srelx, relative=True)

# correlated components to construct covariance matrix
  if xabscor is not None:
    if len(np.shape(np.array(xabscor))) <2:
      dat.add_error(axis='x', err_val=xabscor, correlation=1.)
    else:
      for c in xabscor:
        dat.add_error(axis='x', err_val=c, correlation=1.)
  if yabscor is not None:
    if len(np.shape(np.array(yabscor))) < 2:
      dat.add_error(axis='y', err_val=yabscor, correlation=1.)
    else:
      for c in yabscor:
        dat.add_error(axis='y', err_val=c, correlation=1.)
  if xrelcor is not None:
    if len(np.shape(np.array(xrelcor))) < 2:
      dat.add_error(axis='x', err_val=xrelcor, correlation=1., relative=True)
    else:
      for c in xrelcor:
        dat.add_error(axis='x', err_val=c, correlation=1., relative=True)
        
  # set up fit object
  fit = Fit(dat, func)
  # text for labeling       
  fit.assign_model_function_latex_name(model_name)
  fit.assign_model_function_latex_expression(model_expression)
  fit.model_label = model_legend
  
  # finally, add relative errors with reference to model
  #   - do this here, because this needs methods of fit object 
  if ref_to_model == True:
    ref='model'
  else:
    ref='data'
  if srely is not None: 
    fit.add_error(axis='y', err_val=srely,
                    relative=True, reference=ref)
  if yrelcor is not None:
    if len(np.shape(np.array(yrelcor))) < 2:
      fit.add_error(axis='y', err_val=yrelcor, correlation=1.,
                    relative=True, reference=ref)
    else:
     for c in yrelcor:
       fit.add_error(axis='y', err_val=c, correlation=1.,
                     relative=True, reference=ref)

  # initialize and run fit
  if p0 is not None:
    fit.set_all_parameter_values(p0)

  if dp0 is not None:
    fit.parameter_errors = dp0

  if constraints is not None:
    if not (isinstance(constraints[0], tuple) or isinstance(constraints[0], list)):
      constraints = (constraints,)
    for c in constraints:
      fit.add_parameter_constraint(*c)

  if limits is not None:
    if isinstance(limits[1], list):
      for l in limits:
        fit.limit_parameter(l[0], l[1], l[2])          
    else: 
      fit.limit_parameter(limits[0], limits[1], limits[2])          

  fit.do_fit()                        

# harvest results
#  par, perr, cov, chi2 = fit.get_results() # for kafe vers. > 1.1.0
  parn = np.array(fit.parameter_names) 
  parv = np.array(fit.parameter_values) 
  pare = np.array(fit.parameter_errors)
  cor = np.array(fit.parameter_cor_mat)
  chi2 = fit.goodness_of_fit
  if asym_parerrs:
    parae = np.array(fit.asymmetric_parameter_errors)
  else:
    parae = np.array(list(zip(-pare, pare)))

  if not quiet:
    fit.report(asymmetric_parameter_errors=True)

  if plot:
   # plot data, uncertainties, model line and model uncertainties
    kplot=Plot(fit)
    # set some 'nice' options
    kplot.customize('data', 'marker', ['o'])
    kplot.customize('data', 'markersize', [6])
    kplot.customize('data', 'color', ['darkblue'])
    kplot.customize('model_line', 'color', ['darkorange'])
    kplot.customize('model_line', 'linestyle', ['--'])
    if not plot_band:
      kplot.customize('model_error_band', 'hide', [True])
    else:
      kplot.customize('model_error_band', 'color', ['green'])
      kplot.customize('model_error_band', 'label', [model_band])
      kplot.customize('model_error_band', 'alpha', [0.1])     

    # plot with defined options
    kplot.plot(fit_info=fit_info, asymmetric_parameter_errors=True)

    if plot_cor:
      cpf = ContoursProfiler(fit)
      cpf.plot_profiles_contours_matrix() # plot profile likelihood and contours

    if showplots: plt.show()    
      
  return parv, parae, cor, chi2

def k2hFit(fitf, data, bin_edges, 
           p0 = None, dp0 = None,  constraints = None,
           fixPars=None, limits=None,
           use_GaussApprox = False,
           fit_density = True,
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

      * p0: array-like, initial guess of parameter values
      * dp0: array-like, initial guess of parameter uncertainties
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
  from kafe2.fit.histogram import HistCostFunction_GaussApproximation
  
  # create a data container from input
  nbins = len(bin_edges)-1
  bin_range = (bin_edges[0], bin_edges[-1])
  hdat = HistContainer(nbins, bin_range, bin_edges=bin_edges, fill_data=data)

  # set up fit object
  if use_GaussApprox:
    ##print('Gauss Approx. for histogram data not yet implemented - exiting!')
    hfit = Fit(hdat, fitf,
                cost_function=HistCostFunction_GaussApproximation(
                  errors_to_use='pointwise'),
                  density=fit_density )
  else:   
     hfit = Fit(hdat, fitf,
                cost_function=HistCostFunction_NegLogLikelihood(
                  data_point_distribution='poisson'),
                  density=fit_density )
  # text for labeling       
  hfit.assign_model_function_latex_name(model_name)
  hfit.assign_model_function_latex_expression(model_expression)
  hfit.model_label = model_legend

  # - provide text for labeling ...      
  hdat.label = data_legend
  hdat.axis_labels = axis_labels

  # initialize and run fit
  if p0 is not None:
    hfit.set_all_parameter_values(p0)

  if dp0 is not None:
    hfit.parameter_errors = dp0

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


def xyFit(*args, **kwargs):
  """call xyFit from .phyFit"""
  # print('Calling xyFit() from package phyFit')
  from .phyFit import xyFit 
  return xyFit(*args, **kwargs)
  
def hFit(*args, **kwargs):
  """call hFit from .phyFit"""
  # print('Calling hFit() from package phyFit')
  from .phyFit import hFit 
  return hFit(*args, **kwargs)

def mFit(*args, **kwargs):
  """call mFit from .phyFit"""
  # print('Calling mFit() from package phyFit')
  from .phyFit import mFit 
  return mFit(*args, **kwargs)

def xFit(*args, **kwargs):
  """call xFit from .phyFit"""
  # print('Calling xFit() from package phyFit')
  from .phyFit import xFit 
  return xFit(*args, **kwargs)


## ------- section 6: simulated data -------------------------

def smearData(d, s, srel=None, abscor=None, relcor=None):
  """Generate measurement data from "true" input 
  by adding random deviations according to the uncertainties 

  Args:
    * d:  np-array, (true) input data
    
  the following are single floats or arrays of length of array d
    * s: Gaussian uncertainty(ies) (absolute)
    * srel: Gaussian uncertainties (relative)

  the following are common (correlated) systematic uncertainties
    * abscor: 1d np-array of floats or list of np-arrays:
      absolute correlated uncertainties
    * relcor: 1d np-array of floats or list of np-arrays:
      relative correlated uncertainties

  Returns:
    * np-array of floats: dm, smeared (=measured) data    
  """

  dm = d + s*np.random.randn(len(d)) # add independent (statistical) deviations

  if(srel is not None): 
    dm += d * srel * np.random.randn(len(d)) # add relative  deviations

  if(abscor is not None):
    ac = np.asarray(abscor)
    if len(np.shape(ac)) < 2 : 
      dm += ac * np.random.randn(1) # add common absolute deviation
    else:  # got several components
      for _ac in ac:   # add all common absolute deviations
        dm += _ac * np.random.randn(1) 
        
  if(relcor is not None): 
    rc = np.asarray(relcor)
    if len(np.shape(rc)) < 2 : 
      dm += d * rc * np.random.randn(1) # add common relative  deviation
    else: # got several components
      for _rc in rc:      # add all common relative  deviations
        dm += d * _rc * np.random.randn(1) 

  return dm

def generateXYdata(xdata, model, sx, sy, mpar=None,
   srelx=None, srely=None,
   xabscor=None, yabscor=None, xrelcor=None, yrelcor=None):
  ''' Generate measurement data according to some model
    assumes xdata is measured within the given uncertainties; 
    the model function is evaluated at the assumed "true" values 
    xtrue, and a sample of simulated measurements is obtained by 
    adding random deviations according to the uncertainties given 
    as arguments.

    Args:
      * xdata:  np-array, x-data (independent data)
      * model: function that returns (true) model data (y-dat) for input x
      * mpar: list of parameters for model (if any)
    the following are single floats or arrays of length of x
      * sx: gaussian uncertainty(ies) on x      
      * sy: gaussian uncertainty(ies) on y
      * srelx: relative Gaussian uncertainty(ies) on x      
      * srely: relative Gaussian uncertainty(ies) on y
    the following are common (correlated) systematic uncertainties
      * xabscor: absolute, correlated error on x
      * yabscor: absolute, correlated error on y
      * xrelcor: relative, correlated error on x
      * yrelcor: relative, correlated error on y
    Returns:
      * np-arrays of floats: 

        * xtrue: true x-values
        * ytrue: true value = model(xtrue)
        * ydata:  simulated data  
  '''

  # first, add random statistical and systematic deviations on x
  xtrue = smearData(xdata, sx, srel=srelx, abscor=xabscor, relcor=xrelcor) 
     # take as "true" x

  #calculate model prediction for these:
  if mpar is not None:
   ytrue = model(xtrue, *mpar)
  else:
   ytrue = model(xtrue)

  # add uncertainties to y
  ydata = smearData(ytrue, sy, srel=srely, abscor=yabscor, relcor=yrelcor) 

  return xtrue, ytrue, ydata


## ------- section 7: frequently used helper functions --------------------

def check_function_code(code_string):
  """Check Python code before using it in exec() command
    
   Watch out for "dangerous" actions

    Args: 
     - user-defined code
    Returns:
     - function name
     - code
  """

  FORBIDDEN_TOKENS = ['import', 'exec', 'global', 'execfile']

  for s in FORBIDDEN_TOKENS:
    contains_forbidden_token = False
    if code_string.find(s) >=0:
      _e = "!!! Encountered forbidden token '%s' in user-entered code" % (s)
      contains_forbidden_token = True
    if(contains_forbidden_token):
      raise ValueError(e)

  function_name=''  
  words_in_code = code_string.split()
  for i, w in enumerate(words_in_code):
    if w == 'def':
      fn = words_in_code[i+1]
      function_name=fn[0:fn.find( '(' )]
      break
  if function_name is '':
      raise ValueError(
        "No function name in user entered code.")         
  return function_name, code_string 

def plot_xy_from_yaml(d):
  """plot (xy) data from yaml file

     Input: 

         dictionary from yaml input
     
     Output: 

         matplotlib figure

  yaml-format of input:

  .. code-block:: yaml

     title: <title of plot>
     x_label: <label for x-axis>
     y_label: <label for y-axis>

     label: <name of data set>
     x_data: [ x values ]
     y_data: [ y values ]
     x_errors: x-uncertainty or [x-uncertainties]
     y_errors: y-uncertainty or [y-uncertainties]

     several input sets to be separated by 
     ...
     ---   

  In case a model function is also supplied, it is overlayed in the 
  output graph. The corresponding *yaml* block looks as follows:

  .. code-block:: yaml

    # optional model specification
    model_label: <model name>
    model_function: |
    <Python code of model function>
  """

  import numpy as np, matplotlib.pyplot as plt
  from PhyPraKit import check_function_code
  
  def plot_xy(x, y, ex, ey, title=None,
           label='data', x_label = 'x', y_label = 'y',
           marker='x', color='grey'):
    """return figure with (x,y) data and uncertainties
    """
    # draw data
    plt.plot(x, y, marker=marker, linestyle='', color=color, alpha=0.5)
    if (ex is not None) and (ey is not None):
      plt.errorbar(x, y, xerr=ex, yerr=ey, fmt='.', label=label)
    elif ey is not None:
      plt.errorbar(x, y, yerr=ey, fmt='.', label=label)
    elif ex is not None:
      plt.errorbar(x, y, xerr=ex, fmt='.', label=label)
    else:      
      plt.errorbar(x, y, ey, fmt=".", label=label)

    if x_label is not None: plt.xlabel(x_label, size='x-large')
    if y_label is not None: plt.ylabel(y_label, size='x-large')
    if title is not None:
      plt.title(title, size='xx-large')
    return plt.gcf()
  # -- end plot function 

  if 'title' in d:
    title = d['title']
  else:
    title = None

  if 'label' in d:
    data_label = d['label']
  else:
    data_label = None

  x_dat = list(map(float, d['x_data']))
  if 'x_label' in d:
    x_label = d['x_label']
  else:
    x_label = None
  if 'x_errors' in d:
    e = d['x_errors']
    if type(e) is type([]):
      x_err = list(map(float, e))
    else:
      x_err = float(e)
  else:
    x_err = None

  if 'y_data' in d:  
    y_dat = list(map(float, d['y_data']))
    if 'y_errors' in d:
      e = d['y_errors']
      if type(e) is type([]):
        y_err = list(map(float, e))
      else:
        y_err = float(e)
    else:
      y_err = None
  
  if 'y_label' in d:
    y_label = d['y_label']
  else:
    y_label = None

  if 'y_data' in d:
    fig = plot_xy(x_dat, y_dat, x_err, y_err, title=title,
             label=data_label, x_label = x_label, y_label = y_label,
             marker='x', color='grey')

  # check if model function supplied
  if 'model_function' in d:
    code_str = None
    if 'model_label' in d:
      model_label = d['model_label']
    else:
       model_label = ''
    try:
      code_str = d['model_function']['python_code']
    except:
      try:
        code_str = d['model_function']
      except:
        pass      
    if code_str is not None:
      # check and execute provide model code
      functionName, code = check_function_code(code_str)
      scope = dict()
      header= 'import numpy as np\n' + 'import scipy\n' 
      exec(header + code, scope)
      # plot function 
      xp = np.linspace(min(x_dat), max(x_dat), 113)
      plt.plot(xp, scope[functionName](xp) , '-', label=model_label)

# finally, plot legend of all shown graphs      
  plt.legend(loc='best')

def csv2yaml(file, nlhead=1, delim='\t'):
  """read floating point data in general csv format and convert to yaml  

  skip header lines, replace decimal comma, remove special characters,
  and ouput as yaml data block
  
  Args:
    * file: file name or open file handler
    * nhead: number of header lines; keys taken from first header line
    * delim: column separator

  Returns: 
    * hlines: list of string, header lines
    * ymltxt: list of text lines, each with yaml key and data

  """
  
  # --------------------------------------------------------------------

  import yaml
  
  # -- helper function to filter input lines
  def specialCharFilter(f, delim):
    """a generator to filter lines read from file
    replace German ',' by '.', remove special characters 

    Args:
      * string f:  file name
    Yields:
      * a valid line with numerical data
    """
    while True:
      l=f.readline()      # read one line
      if (not l): break   # end-of-file reached, exit

    # remove white spaces and control characters, fix German floats 
        # remove leading and trailing white spaces
      l=l.strip()         
      # remove ascii control characters (except delimiter) 
      for i in range(32):
        if delim != chr(i) : l=l.replace(chr(i),'') 
      if l=='': continue        # skip empty lines 
        # replace German decimal comma (if not CSV format)
      if delim != ',' :
        filtline=l.replace(',', '.')
      else:
        filtline=l      
      yield filtline           # pass filtered line to loadtxt()
#   -- end specialCharFilter

  # open file for read (if necessary)
  if type(file)==type(' '): f = open(file, 'r') # file is a file name
  else: f = file        # assume input is file handle of an open file 

  # set-up generator for text lines from file 
  lfilt = specialCharFilter(f, delim)

  # read header
  hlines=[]
  for i in range (nlhead):
    hlines.append(next(lfilt))  # header line(s)

  dlines=[]
  while True:
    try:
      dlines.append(next(lfilt).split(delim))  # data line(s)
    except StopIteration:
      break
    
  Nlin = len(dlines)
  Ncol = len(dlines[0])
  #print(" --> number of columns", Ncol)
  #print(" --> number of data points", Nlin)

  # interpret strings in header[0] as keys
  keys = hlines[0].split(delim)
  Nkey = len(keys)
  if Nkey != Ncol:
    print('Nkey=', Nkey, ' Ncol=', Ncol)
    raise ValueError(
      "!!! number of keys{} not equal number of columns - exit")

  # construct string in yaml format
  #   transpose list with number strings
  dlinesT = [[float(row[i]) for row in dlines] for i in range(Ncol)]
#  dlinesT = [[row[i] for row in dlines] for i in range(Ncol)]
  ylines=[]
  for i, k in enumerate(keys):
    #yl = "{0}: [{1}]".format(k.strip(), ','.join(dlinesT[i]))
    yl = yaml.dump({ k.strip() : dlinesT[i] },
                     default_flow_style=True ) # arrays as [ ... ]
    yl = yl.replace('{','')  # remove { and }
    yl = yl.replace('}','')
    ylines.append(yl)
  return hlines, ylines

def plot_hist_from_yaml(d):
  """plot histogram data from yaml file

     Input: 

         dictionary from yaml input
     
     Output: 

         matplotlib figure

  yaml-format of input:

  .. code-block:: yaml

     title: <title of plot>
     x_label: <label for x-axis>
     y_label: <label for y-axis>

     label: <name of data set>
     raw_data: [x1, ... , xn]
     # define binning
     n_bins: n
     bin_range: [x_min, x_max]
     #   alternatively: 
     # bin edges: [e0, ..., e_n]

     several input sets to be separated by 
     ...
     ---   
  """
  import numpy as np, matplotlib.pyplot as plt
  from PhyPraKit import histstat, check_function_code

  # trick to generate a global variable for accumulated statistics
  global statinfo
  try: statinfo
  except NameError: statinfo = []

  def plot_hist(bconts, bedges, title=None,
           label='histogram', x_label = 'x', y_label = 'y',
           grid=True, statistics=True):
    """return figure with histogram data
    """
    # global variable for statistics information
    global statinfo

    # draw data
    w = 0.9*(be[1:] - be[:-1])
    
    plt.bar(bedges[:-1],bconts,
            align='edge', width = w, alpha=0.5,
#            facecolor='cadetblue',
            edgecolor='grey', 
            label = label)
    
    # get statistical information
    if statistics:
      mean, sigma, sigma_m = histstat(bconts, bedges, pr=False)
      if len(statinfo) == 0:
        statinfo.append('Statistics:')
      else:      
        statinfo.append('  - - - - - - - ')
      statinfo.append('  $<>$:  {:.3g}'.format(mean))
      statinfo.append('     $\sigma$   : {:.3g}'.format(sigma))
    
    if x_label is not None: plt.xlabel(x_label, size='x-large')
    if y_label is not None: plt.ylabel(y_label, size='x-large')
    if title is not None:
      plt.title(title, size='xx-large')
    if grid: plt.grid()
    return plt.gcf()
  # -- end plot function 

  # get data
  if 'raw_data' in d:
    hdata = list(map(float, d['raw_data']))
  bins = 10
  if 'n_bins' in d:
    bins = d['n_bins']
  if 'bin_edges' in d:
    bins = d['bin_edges']
    xmn = bins[0]
    xmx = bins[-1]
  bin_range = None
  if 'bin_range' in d:
    bin_range = d['bin_range']
    xmn = bin_range[0]
    xmx = bin_range[-1]
    
  if 'title' in d:
    title = d['title']
  else:
    title = None

  if 'label' in d:
    data_label = d['label']
  else:
    data_label = None
  
  if 'x_label' in d:
    x_label = d['x_label']
  else:
    x_label = 'x'
  if 'y_label' in d:
    y_label = d['y_label']
  else:
    y_label = 'y'

  if 'raw_data' in d:
    bc, be = np.histogram(hdata, bins=bins,  range=bin_range)  
    fig = plot_hist(bc, be, title=title,
             label=data_label, x_label = x_label, y_label = y_label,
             grid=True)
    xmn = be[0]
    xmx = be[-1]
    sfac = bc.sum() * (be[1]-be[0])
  else:
    sfac = 1. 
    
  # check if model function supplied
  if 'model_density_function' in d :
    code_str = None
    if 'model_label' in d:
      model_label = d['model_label']
    else:
       model_label = ''
    try:
      code_str = d['model_density_function']['python_code']
    except:
      try:
        code_str = d['model_density_function']
      except:
        pass      
    if code_str is not None:
      # check and execute provide model code
      functionName, code = check_function_code(code_str)
      scope = dict()
      header= 'import numpy as np\n' + 'import scipy\n' 
      exec(header + code, scope)
      # plot function 
      xp = np.linspace(xmn, xmx, 113)
      ## sfac = bc.sum() * (be[1]-be[0])
      plt.plot(xp, sfac*scope[functionName](xp) , '-', label=model_label)

# finally, plot legend of all shown graphs
  plt.legend(loc='best', title="\n".join(statinfo))
