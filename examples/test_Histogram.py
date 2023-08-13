#
"""test_Historgram.py
   demonstrate histogram functionality in PhyPraKit

.. moduleauthor:: Guenter Quast <g.quast@kit.edu>

"""

import numpy as np, matplotlib.pyplot as plt
from PhyPraKit import histstat, hist2dstat, profile2d, chi2p_indep2d


## ------- tools to generate test data  ----------------


#  some examples to generate 2d distributions
def genXYdata(isize, function):
    return function(size=isize)


def gen2dgaus(size=10000, mux=5.0, sigx=10.0, muy=20.0, sigy=3.0, rho=0.15):
    # generate two arrays with pairs of correlated gaussian numbers
    u = np.random.randn(size)
    x = mux + sigx * u  # gauss, with mean mux and sigma sigx
    v = np.random.randn(size)
    y = muy + sigy * (rho * u + np.sqrt(1.0 - rho**2) * v)
    return x, y


def gen2dgaus(size=10000, mu=5.0, sig=1.0, tau=1.0, rho=0.1):
    # generate two arrays with pairs random numbers,
    #  gauss in y , exp. in x, correlated
    x = -tau * np.log(np.random.rand(size))
    y = mu + sig * np.random.randn(size) + rho * x
    return x, y


# ---------------------------------------------------------------
if __name__ == "__main__":
    # demonstrate usage of histogram functionality in package PhyPraKit

    fig = plt.figure(1, figsize=(10.0, 10.0))
    x, y = genXYdata(10000, gen2dgaus)
    nbinsx = 20
    nbinsy = 25

    ax_y = plt.subplot(2, 2, 1)  # y histogram and statistics
    #  bincont,binedge = nhist(y,nbinsy,xlabel="y") # histogram data
    bcy, bey, p = plt.hist(y, nbinsy)  # histogram data
    ax_y.set_xlabel("y")
    ax_y.set_ylabel("frequency")
    histstat(bcy, bey)

    ax_x = plt.subplot(2, 2, 4)  # x histogram and statistics
    #  bincont,binedge = nhist(x,nbinsx) # histogram data
    bcx, bex, p = plt.hist(x, nbinsx)  # histogram data
    ax_x.set_xlabel("x")
    ax_x.set_ylabel("frequency")
    histstat(bcx, bex)

    ax_xy = plt.subplot(2, 2, 2)  # 2d historgram and statistics
    #  H, xedges, yedges = nhist2d(x,y,[nbinsx,nbinsy],'var 1', 'var 2')
    H, xedges, yedges, p = plt.hist2d(x, y, [nbinsx, nbinsy], cmap="Blues")
    ax_xy.set_xlabel("x")
    ax_xy.set_ylabel("y")
    hist2dstat(H, xedges, yedges, True)
    # print p-value of chi2 independence test
    chi2p_indep2d(H, bcx, bcy, True)

    # make a "Profile Plot" - mean y at a given x, standard deviation as error bar
    ax_profile = plt.subplot(2, 2, 3)  # profile historgram
    profile2d(H, xedges, yedges)
    ax_profile.set_xlabel("x", size="x-large")
    ax_profile.set_ylabel("mean of y", size="x-large")

    plt.show()
