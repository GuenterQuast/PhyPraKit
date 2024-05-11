#! /usr/bin/env python3
"""test_xFit.py
   fit to indexed data x_i with iminiut

.. moduleauthor:: Guenter Quast <g.quast@kit.edu>

"""

import numpy as np, matplotlib.pyplot as plt

##from PhyPraKit.phyFit import xFit
from PhyPraKit import xFit

if __name__ == "__main__":  # --------------------------------------
    #
    # *** Example of an application of phyFit.xFit() to fit indexed data
    #
    #     Coordinates in r-phi are averaged and transformed to cartesian

    def cartesian_to_polar(data, x=1.0, y=1.0):
        # determine polar coordinats from cartesian (x,y)
        nm = len(data) // 2  # expect two arrays with measurements
        r = np.sqrt(x * x + y * y) * np.ones(nm)
        phi = np.arctan2(y, x) * np.ones(nm)
        return np.concatenate((r, phi))

    #
    # fit input example: (r, phi) of two space points in polar coordinates
    p_vals = np.array([0.902, 0.873, 0.73, 0.79])
    p_u = np.array([0.0, 0.0, 0.16, 0.13])
    p_ru = np.array([0.015, 0.015, 0.0, 0.0])
    p_cru = np.array([0.01, 0.01, 0.0, 0.0])

    #
    # perform fit to data with function xFit using class mnFit
    resultDict = xFit(
        cartesian_to_polar,
        p_vals,
        s=p_u,
        srel=p_ru,
        sabscor=None,
        srelcor=p_cru,
        names=["r", "r", r"$\varphi$", r"$\varphi$"],
        # p0=(1., 1.),
        use_negLogL=True,
        plot=True,
        plot_band=True,
        plot_cor=True,
        showplots=False,
        quiet=False,
        axis_labels=["Index", "x   |  x(*par)"],
        data_legend="Polar Data",
        model_legend="r-phi from x-y",
    )
    plt.suptitle(
        "mnFit example: fit to indexed data", size="xx-large", color="darkblue"
    )

    # Print results
    pvals, perrs, cor, chi2, pnams = resultDict.values()
    print("\n*==* xyFit Result:")
    print(" parameter names:       ", pnams)
    print(" chi2: {:.3g}".format(chi2))
    print(" parameter values:      ", pvals)
    print(" neg. parameter errors: ", perrs[:, 0])
    print(" pos. parameter errors: ", perrs[:, 1])
    print(" correlations : \n", cor)

    plt.show()
