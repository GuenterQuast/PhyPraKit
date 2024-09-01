"""test_convolutionFilter.py
   Read data exported with PicoScope usb-oscilloscpe,
   here the accoustic excitation of a steel rod

   Demonstrates usage of convolutionFilter for detection
   of signal maxima and falling edges

.. moduleauthor:: Guenter Quast <g.quast@kit.edu>

"""

# -----example Code illustrating usage --------------------
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import PhyPraKit as ppk
    import sys

    # check for / read command line arguments
    if len(sys.argv) == 2:
        fname = sys.argv[1]
    else:
        fname = "Wellenform.csv"
    print(
        "\n*==* script " + sys.argv[0] + " executing \n",
        "     processing file " + fname,
    )

    # read data from PicoScope
    units, data = ppk.readPicoScope(fname, prlevel=2)
    t = data[0]
    a = data[1]

    # run peak and edge finder
    width = 30
    #  use convoluted template filter
    pidx = ppk.convolutionPeakfinder(a, width, th=0.8)
    didx = ppk.convolutionEdgefinder(-a, width, th=0.4)
    if len(pidx) > 3:
        print(" --> %i peaks and %i edges found" % (len(pidx), len(didx)))
        tp, ap = np.array(t[pidx]), np.array(a[pidx])
        td, ad = np.array(t[didx]), np.array(a[didx])
    else:
        print("*!!* not enough peaks found - tune peakfinder parameters!")
        sys.exit(1)

    # Filter peaks and dips:  keep only largest ones
    #    !!! need inspection by eye to ensure correct peaks are identified
    tpm = []
    apm = []
    for i, ti in enumerate(tp):
        if ap[i] > 0.133:
            tpm.append(tp[i])
            apm.append(ap[i])
    tpm = np.array(tpm)
    apm = np.array(apm)

    tdm = []
    adm = []
    for i, ti in enumerate(td):
        if ad[i] < -0.07:
            tdm.append(td[i])
            adm.append(ad[i])
    tdm = np.array(tdm)
    adm = np.array(adm)

    print(" --> %i large peaks and %i large dips found" % (len(tpm), len(tdm)))

    # make a plots
    fig = plt.figure(1, figsize=(7.5, 7.5))
    fig.suptitle("Script: test_convolutionFilter.py", size="x-large", color="b")
    fig.subplots_adjust(left=0.14, bottom=0.1, right=0.97, top=0.93, wspace=None, hspace=0.25)  #
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(tpm, apm, "rx", alpha=0.9, label="large peaks")
    ax1.plot(tdm, adm, "mx", alpha=0.9, label="falling edges")
    ax1.plot(t, a)
    ax1.set_xlabel("$time$ " + units[0], size="large")
    ax1.set_ylabel("$Amplitude$ " + units[1], size="large")
    ax1.legend(loc="best", numpoints=1, prop={"size": 10})
    ax1.grid()

    # plot distribution of time differences between peaks/dips
    dtp = tpm[1:] - tpm[:-1]
    dtd = tdm[1:] - tdm[:-1]
    ax2 = fig.add_subplot(2, 1, 2)
    bins = np.linspace(min(min(dtp), min(dtd)), max(max(dtp), max(dtd)), 50)
    bc, be, _ = ax2.hist(
        [dtp, dtd],
        bins,
        stacked=True,
        color=["r", "m"],
        label=["peaks", "edges"],
        alpha=0.5,
    )
    ax2.set_xlabel(r"$Zeitdifferenz\,der\,peaks / dips$ (ms)", size="large")
    ax2.legend(loc="best", numpoints=1, prop={"size": 10})
    ax2.set_ylabel(r"$H\"aufigkeit$", size="large")
    ax2.grid()

    print("** Histogram statistics:")
    m_dtp, s_dtp, sm_dtp = ppk.histstat(bc[0], be, pr=False)
    m_dtd, s_dtd, sm_dtd = ppk.histstat(bc[1], be, pr=False)
    print(" --> mean time differnce of   peaks: (%.5g +/- %.2g) ms" % (m_dtp, sm_dtp))
    print("                              dips:  (%.5g +/- %.2g) ms" % (m_dtd, sm_dtp))
    ax2.text(
        0.1,
        0.85,
        r"peaks: (%.5g$\pm$%.2g) ms" % (m_dtp, sm_dtp),
        transform=ax2.transAxes,
    )
    ax2.text(
        0.1,
        0.75,
        r" edges: (%.5g$\pm$%.2g) ms" % (m_dtd, sm_dtd),
        transform=ax2.transAxes,
    )

    plt.show()
