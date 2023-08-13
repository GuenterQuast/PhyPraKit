#! /usr/bin/env python3
"""test_odFit
   test fitting an arbitrary fucntion with scipy odr, 
   with uncertainties in x and y 

.. moduleauthor:: Guenter Quast <g.quast@kit.edu>
"""

from PhyPraKit import generateXYdata, odFit
import numpy as np, matplotlib.pyplot as plt


# -- the model function
def model(x, a=0.3, b=1.0, c=1.0):
    return a * x**2 + b * x + c


if __name__ == "__main__":  # --------------------------------------
    # parameters for the generation of test data
    sigx_abs = 0.2  # absolute error on x
    sigy_abs = 0.1  # relative error on y
    xmin = 1.0
    xmax = 10.0
    xdata = np.arange(xmin, xmax + 1.0, 1.0)
    nd = len(xdata)
    mpars = [0.3, -1.5, 0.5]

    # generate the data
    xt, yt, ydata = generateXYdata(xdata, model, sigx_abs, sigy_abs, mpar=mpars)

    # fit with odFit (uses scipy.curve_fit and scipy.odr)
    par, pare, cor, chi2 = odFit(
        model, xdata, ydata, sigx_abs, sigy_abs, p0=None  # data and uncertaintites
    )

    # setting any of the above names to None will remove the entry from the legend,
    #  if not specified, use default

    print("*==* data set")
    print("  x = ", xdata)
    print("  sx = ", sigx_abs)
    print("  y = ", ydata)
    print("  sy = ", sigy_abs)
    print("*==* fit result:")
    print("  -> chi2:         %.3g" % chi2)
    np.set_printoptions(precision=3)
    print("  -> parameters:   ", par)
    np.set_printoptions(precision=2)
    print("  -> uncertainties:", pare)
    print("  -> correlation matrix: \n", cor)
