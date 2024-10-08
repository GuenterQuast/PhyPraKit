#! /usr/bin/env python3
"""test_kRegression
   test linear regression with kafe using kFit from PhyPrakKit
   uncertainties in x and y and correlated
   absolute and relative uncertainties

.. moduleauthor:: Guenter Quast <g.quast@kit.edu>
"""

from PhyPraKit import generateXYdata, kFit
from kafe.function_library import linear_2par
import numpy as np


# -- the model function for data generation
def model(x, a=0.3, b=1.0):
    """simple linear function"""
    return a * x + b


if __name__ == "__main__":  # --------------------------------------
    # set some uncertainties
    sigx_abs = 0.2  # absolute error on x
    sigy_rel = 0.1  # relative error on y
    #       errors of this kind only supported by kafe
    sxrelcor = 0.05  #  a relative, correlated error on x
    syabscor = 0.1  #  an absolute, correlated error on y
    xmin = 1.0
    xmax = 10.0

    xdata = np.arange(xmin, xmax + 1.0, 1.0)
    nd = len(xdata)
    # generate pseudo data
    xt, yt, ydata = generateXYdata(xdata, model, sigx_abs, 0.0, srely=sigy_rel, xrelcor=sxrelcor, yabscor=syabscor)
    ey = sigy_rel * yt * np.ones(nd)  # set array of relative y errors

    # (numerical) linear regression
    par, pare, cor, chi2 = kFit(
        linear_2par,
        xdata,
        ydata,
        sigx_abs,
        ey,
        p0=(1.0, 1.0),  # p0e=(0.2, 0.2),
        xrelcor=sxrelcor,
        yabscor=syabscor,
        plot=True,
        quiet=True,
    )

    print("*==* input data")
    print(("  x = ", xdata))
    print(("  sx = ", sigx_abs))
    print(("  y = ", ydata))
    print(("  sy = ", ey))
    print("fit result:")
    print(
        (
            "  a=%.2f+-%.2f, b=%.2f+-%.2f, corr=%.2f, chi2/df=%.2f\n"
            % (par[0], pare[0], par[1], pare[1], cor[0, 1], chi2 / (nd - 2.0))
        )
    )
