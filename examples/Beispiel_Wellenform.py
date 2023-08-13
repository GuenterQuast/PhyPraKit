# -*- coding: utf-8 -*-
"""Beispiel_Wellenform.py 
   Einlesen von mit PicoScope erstellten Dateien am Beispiel
   der akustischen Anregung eines Stabes

   - Fourier-Analyse des Signals

   - Bestimmung der Resonanzfrequenz mittels Autokorrelation
  
.. moduleauthor:: Guenter Quast <g.quast@kit.edu>

"""

# -----example Code illustrating usage --------------------
if __name__ == "__main__":
    import numpy as np, matplotlib.pyplot as plt, PhyPraKit as ppk
    from scipy import interpolate, signal
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

    print("** Fourier Spectrum")
    freq, amp = ppk.FourierSpectrum(t, a, fmax=20)
    # freq, amp = ppk.Fourier_fft(t, a)  # use fast algorithm
    frequency = freq[np.where(amp == max(amp))]
    print(" --> Frequenz mit max. Amplitude: ", frequency)

    # calculate autocorrelation function
    print("** Autocorrelation Function")
    ac_a = ppk.autocorrelate(a)
    ac_t = t - t[0]

    # run peak finder
    width = 80
    #  use convoluted template filter
    pidx = ppk.convolutionPeakfinder(ac_a, width, th=0.4)
    if len(pidx) > 3:
        print(" --> %i auto-correlation peaks found" % (len(pidx)))
        pidx[0] = 0  # first peak is at 0 by construction
        tp, ap = np.array(ac_t[pidx]), np.array(ac_a[pidx])
    else:
        print("*!!* not enough peaks found - tune peakfinder parameters!")
        sys.exit(1)

    # Filter peaks and dips:  keep only largest ones
    #    !!! need inspection by eye to ensure correct peaks are identified
    tpm = []
    apm = []
    for i, ti in enumerate(tp):
        if ap[i] > 0.137:
            tpm.append(tp[i])
            apm.append(ap[i])
    tpm = np.array(tpm)
    apm = np.array(apm)

    print(" --> %i (large) peaks found" % len(tpm))

    # make a plots
    fig = plt.figure(1, figsize=(10.0, 7.5))
    fig.suptitle("Script: Beispiel_Wellenform.py", size="x-large", color="b")
    fig.subplots_adjust(
        left=0.14, bottom=0.1, right=0.97, top=0.93, wspace=None, hspace=0.25
    )  #
    # Signalverlauf
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(t, a)
    ax1.set_xlabel("$Zeit$ " + units[0], size="large")
    ax1.set_ylabel("$Amplitude$ " + units[1], size="large")
    ax1.grid()
    # Fourier-Spektrum
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(freq, amp, "b-")
    ax2.set_xlabel("$Frequenz$ $f$ (kHz)", size="large")
    ax2.set_ylabel("$Amplitude$", size="large")
    ax2.set_yscale("log")
    ax2.grid()
    # Auto-Korrelation
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(tpm, apm, "rx", alpha=0.9, label="large peaks")
    ax3.plot(ac_t, ac_a)
    ax3.plot([0.0, ac_t[-1]], [1.0, 0.0], "m--", alpha=0.3)  # maximum auto-correlation
    ax3.set_xlabel("$Zeit$ " + units[0], size="large")
    ax3.set_ylabel("$Autocorrelation$ " + units[1], size="large")
    ax3.legend(loc="best", numpoints=1, prop={"size": 10})
    ax3.grid()
    # statistische Auswertung
    # plot distribution of time differences between peaks/dips
    dtp = tpm[1:] - tpm[:-1]
    ax4 = fig.add_subplot(2, 2, 4)
    bins = np.linspace(min(dtp), max(dtp), 50)
    bc, be, _ = ax4.hist(dtp, bins, stacked=True, color="r", label="peaks", alpha=0.5)
    ax4.set_xlabel(r"$Zeitdifferenz\,der\,peaks$ (ms)", size="large")
    #  ax4.legend(loc='best', numpoints=1, prop={'size':10})
    ax4.set_ylabel(r"$H\"aufigkeit$", size="large")
    ax4.grid()

    print("** Histogram statistics:")
    m_dtp, s_dtp, sm_dtp = ppk.histstat(bc, be, pr=False)
    print(" --> mean time differnce of   peaks: (%.5g +/- %.2g) ms" % (m_dtp, sm_dtp))
    ax4.text(
        0.05,
        0.9,
        "mean=(%.5g$\pm$%.2g) ms" % (m_dtp, max(sm_dtp, (be[1] - be[0]) / np.sqrt(12))),
        transform=ax4.transAxes,
    )

    plt.show()
