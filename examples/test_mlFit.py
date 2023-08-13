#! /usr/bin/env python3
"""test_mlFit.py
   Maximum likelihood fit to unbinned data with package phyFit and iminiut

.. moduleauthor:: Guenter Quast <g.quast@kit.edu>

"""

import numpy as np, matplotlib.pyplot as plt

# from PhyPraKit.phyFit import mFit
from PhyPraKit import mFit

if __name__ == "__main__":  # --------------------------------------
    #
    # **unbinned maximum-likelihodd fit** of pdf to unbinned data
    #

    #  real data from measurement with a Water Cherenkov detector ("Kamiokanne")
    #
    #    numbers represent time differences (in µs) between the passage of a muon
    #    and the registration of a second pulse, often caused by an electron from
    #    the decay of the stopped muon. As such events are rare, histogramming the
    #    data prior to fitting would introduce shifts and biases, and therefore the
    #    unbinned fit is the optimal method for this and simular use cases.
    dT = [
        7.42,
        3.773,
        5.968,
        4.924,
        1.468,
        4.664,
        1.745,
        2.144,
        3.836,
        3.132,
        1.568,
        2.352,
        2.132,
        9.381,
        1.484,
        1.181,
        5.004,
        3.06,
        4.582,
        2.076,
        1.88,
        1.337,
        3.092,
        2.265,
        1.208,
        2.753,
        4.457,
        3.499,
        8.192,
        5.101,
        1.572,
        5.152,
        4.181,
        3.52,
        1.344,
        10.29,
        1.152,
        2.348,
        2.228,
        2.172,
        7.448,
        1.108,
        4.344,
        2.042,
        5.088,
        1.02,
        1.051,
        1.987,
        1.935,
        3.773,
        4.092,
        1.628,
        1.688,
        4.502,
        4.687,
        6.755,
        2.56,
        1.208,
        2.649,
        1.012,
        1.73,
        2.164,
        1.728,
        4.646,
        2.916,
        1.101,
        2.54,
        1.02,
        1.176,
        4.716,
        9.671,
        1.692,
        9.292,
        10.72,
        2.164,
        2.084,
        2.616,
        1.584,
        5.236,
        3.663,
        3.624,
        1.051,
        1.544,
        1.496,
        1.883,
        1.92,
        5.968,
        5.89,
        2.896,
        2.76,
        1.475,
        2.644,
        3.6,
        5.324,
        8.361,
        3.052,
        7.703,
        3.83,
        1.444,
        1.343,
        4.736,
        8.7,
        6.192,
        5.796,
        1.4,
        3.392,
        7.808,
        6.344,
        1.884,
        2.332,
        1.76,
        4.344,
        2.988,
        7.44,
        5.804,
        9.5,
        9.904,
        3.196,
        3.012,
        6.056,
        6.328,
        9.064,
        3.068,
        9.352,
        1.936,
        1.08,
        1.984,
        1.792,
        9.384,
        10.15,
        4.756,
        1.52,
        3.912,
        1.712,
        10.57,
        5.304,
        2.968,
        9.632,
        7.116,
        1.212,
        8.532,
        3.000,
        4.792,
        2.512,
        1.352,
        2.168,
        4.344,
        1.316,
        1.468,
        1.152,
        6.024,
        3.272,
        4.96,
        10.16,
        2.14,
        2.856,
        10.01,
        1.232,
        2.668,
        9.176,
    ]

    def modelPDF(t, tau=2.0, fbg=0.2, a=1.0, b=11.5):
        """Probability density function
        for an exponential decay with flat background. The pdf is normed for the
        interval [a=1µs,  b=11.5µs); these parameters a and b must be fixed in the fit!
        """
        pdf1 = np.exp(-t / tau) / tau / (np.exp(-a / tau) - np.exp(-b / tau))
        pdf2 = 1.0 / (b - a)
        return (1 - fbg) * pdf1 + fbg * pdf2

    rdict = mFit(
        modelPDF,
        data=dT,  # data - if not None, normalised PDF is assumed as model
        p0=None,  # initial guess for parameter values
        #  constraints=[['tau', 2.2, 0.01], # Gaussian parameter constraints
        limits=("fbg", 0.0, 1.0),  # paramter limits
        fixPars=["a", "b"],  # fix parameter(s)
        neg2logL=True,  # use  -2 * ln(L)
        plot=True,  # plot data and model
        plot_band=True,  # plot model confidence-band
        plot_cor=False,  # plot profiles likelihood and contours
        showplots=False,  # show / don't show plots
        quiet=False,  # suppress informative printout if True
        axis_labels=[
            "life time  " + "$\Delta$t ($\mu$s)",
            "Probability Density  pdf($\Delta$t; *p)",
        ],
        data_legend="$\mu$ lifetime data",
        model_legend="exponential decay + flat background",
    )

    plt.suptitle(
        "Unbinned ML fit of an exponential + flat distribution",
        size="xx-large",
        color="darkblue",
    )

    # Print results
    #  pvals, perrs, cor, gof, pnams = rdict.values()
    #  print('\n*==* unbinned ML Fit Result:')
    #  print(" parameter names:       ", pnams)
    #  print(" parameter values:      ", pvals)
    #  print(" neg. parameter errors: ", perrs[:,0])
    #  print(" pos. parameter errors: ", perrs[:,1])
    #  print(" correlation matrix : \n", cor)
    ## new version
    print("\n*==* unbinned ML Fit Result:")
    for key in rdict:
        print("{}\n".format(key), rdict[key])

    plt.show()
