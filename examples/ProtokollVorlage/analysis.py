#! /usr/bin/env python3
"""analyisis. py
   simple template analysis with kafe and LaTeX output

   - test fiting an arbitrary fucntion with kafe,
     with uncertainties in x and y and correlated
     absolute and relative uncertainties, then

   - write figure and table to be included in LaTeX document

.. moduleauthor:: Guenter Quast <g.quast@kit.edu>
"""

import kafe  # must be imported first to properly set matplotlib backend
import numpy as np
import matplotlib.pyplot as plt

# some more useful functions from kafe
from kafe.function_tools import FitFunction, LaTeX, ASCII
from kafe.fit import round_to_significance

#
from PhyPraKit import readCSV, writeTexTable


# -- the model function (as in kafe, with decorator functions)
@ASCII(expression="slope * x + y_intercept")
@LaTeX(name="f", parameter_names=("a", "b"), expression="a\\,x+b")
@FitFunction
def model(x, slope=1.0, y_intercept=0.0):
    return slope * x + y_intercept


### --- read data and set errors
hlines, data = readCSV("ToyData.dat")
xdat = data[0]
ydat = data[1]
nd = len(xdat)
sigx_abs = 0.2  # absolute error on x
sigy_rel = 0.1  # relative error on y
#       errors of this kind only supported by kafe
sxrelcor = 0.05  #  a relative, correlated error on x
syabscor = 0.1  #  an absolute, correlated error on y

### --- perform a fit with kafe
#    create the kafe data set ...
dat = kafe.Dataset(data=(xdat, ydat), title="ToyData", axis_labels=["X", "Y"], basename="kRegression")
#    ... and add all error sources
dat.add_error_source("x", "simple", sigx_abs)
dat.add_error_source("x", "simple", sxrelcor, relative=True, correlated=True)
ey = np.absolute(sigy_rel * ydat * np.ones(nd))  # array of relative y errors
dat.add_error_source("y", "simple", ey)
dat.add_error_source("y", "simple", syabscor, correlated=True)
#    set-up the fit ...
fit = kafe.Fit(dat, model)
#    ... run it ...
fit.do_fit(quiet=False)

#   ... harvest results in local variables
par, par_err, cov, chi2 = fit.get_results()  # for kafe vers. > 1.1.0
cor = cov / np.outer(par_err, par_err)

# produce plots
kplot = kafe.Plot(fit)
kplot.plot_all()
# kplot.show() #
plt.draw()
plt.pause(2.0)  # show plot for 2s.

# save input data as table (in include-direcotory for LaTeX)
data = np.array([xdat, sigx_abs * np.ones(nd), ydat, ey])
if writeTexTable(
    "analysis/Table1.tex",
    data,
    cnames=["X", "$\\sigma_X$", "Y", "$\\sigma_Y$"],
    caption="ToyData; außer den in der Tabelle "
    + "angegbenen Unsicherheiten gibt es noch eine "
    + "gemeinsame Unsicherheit von "
    + str(syabscor)
    + " auf die Y-Werte und eine gemeinsame relative"
    + " Unsicherheit von "
    + str(sxrelcor * 100)
    + "\\% auf die X-Werte.",
    fmt="%.4g",
):
    print("Error from writeTexTable ")

# save kafe Figure (in include-directory for LaTeX)
kplot.figure.savefig("analysis/Figure1.pdf")

# finally, summarize and print results
#   first, round to reasonable number of digits
lp = len(par)
p = [0.0] * lp
pe = [0.0] * lp
for i in range(lp):
    p[i], pe[i] = round_to_significance(par[i], par_err[i])

print("*==* data set")
print("  x = ", xdat)
print("  sx = ", sigx_abs)
print("  y = ", ydat)
print("  sy = ", ey)
print("*==* fit result:")
print("  -> chi2:         %.3g" % chi2)
# np.set_printoptions(precision=3)
print("  -> parameters:   ", p)
print("  -> uncertainties:", pe)
np.set_printoptions(precision=2)
print("  -> correlation matrix:\n", cor)
