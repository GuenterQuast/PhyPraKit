#! /usr/bin/env python3
"""test_simplek2Fit

   test fitting simple line with kafe2, without any errors given

.. moduleauthor:: Guenter Quast <g.quast@kit.edu>
"""

# import kafe2 # must be imported first to properly set matplotlib backend
from PhyPraKit import k2Fit
import numpy as np


# -- the model function
def model(x, a=1.0, b=0):
    return a * x + b


if __name__ == "__main__":  # --------------------------------------
    # parameters for the generation of test data
    xmin = 1.0
    xmax = 10.0
    xdata = np.arange(xmin, xmax + 1.0, 1.0)
    ydata = [1.1, 1.9, 2.95, 4.1, 4.9, 6.2, 6.85, 8.05, 8.9, 10.15]
    ey = None
    # fit with kafe2
    par, pare, cor, chi2 = k2Fit(model, xdata, ydata, sy=ey)

    print("*==* data set")
    print("  x = ", xdata)
    print("  y = ", ydata)
    print("  sy = ", ey)
    print("*==* fit result:")
    print("  -> chi2:         %.3g" % chi2)
    np.set_printoptions(precision=3)
    print("  -> parameters:   ", par)
    np.set_printoptions(precision=2)
    print("  -> uncertainties:", pare)
    print("  -> correlation matrix: \n", cor)
