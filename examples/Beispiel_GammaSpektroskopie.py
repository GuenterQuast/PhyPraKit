# -*- coding: utf-8 -*-
"""Beispiel_GammaSpektroskopie.py
  Darstellung der Daten aus einer im CASSY labx-Format gespeicherten Datei
  am Beispiel der Gamma-Spektroskopie

  * Einlesen der Daten im .labx oder gezippten .labx-Format

  Args:

    - name of file in .labx format
    - flag for file type:  0: text, 1: zipped

.. moduleauthor:: Guenter Quast <g.quast@kit.edu>

"""

# last changed: Jan -2024

# -----------------------------------------------------------
import numpy as np
import sys

# -----example Code illustrating usage --------------------
if __name__ == "__main__":
    import sys
    import numpy as np
    import matplotlib.pyplot as plt
    from PhyPraKit import labxParser

    unzip = False
    # check for / read command line arguments
    if len(sys.argv) >= 2:
        fname = sys.argv[1]  # file name
        if len(sys.argv) == 3:  # 0: do not unzip, 1: unzip input file
            unzip = sys.argv[2]
    else:
        fname = "GammaSpektra.labx"
    names, values = labxParser(fname, prlevel=0, unzip=unzip)

    #
    k = []
    n = []
    # collect data we are interested in:
    print("\n *==* Data received:")
    for i, tag in enumerate(names):
        print((tag.split(":"), "length = ", len(values[i][:])))
        tnam = tag.split(":")[1]
        if tnam == "Kanal":
            k.append(np.array(values[i][:]))
        if tnam == "Ereignisse":
            n.append(np.array(values[i][:]))
    # not claer what these are ...
    #    if tnam=='Zeit':   t = np.array(values[i][:])
    #    if tnam=='Spannung':  U = np.array(values[i][:])

    # define a Figure
    fig = plt.figure("Gammaspektren", figsize=(10.0, 7.5))
    fig.suptitle("Script Beispiel_GammaSpektroskopie.py", size="x-large", color="b")
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.97, top=0.93, wspace=None, hspace=0.25)  #
    # define subplots
    ax1 = fig.add_subplot(1, 1, 1)
    for i in range(len(k)):
        ax1.plot(k[i], n[i], label="Spektrum #" + str(i))
    ax1.set_xlabel("Kanal Nummer", size="large")
    ax1.set_ylabel("Anzahl", size="large")
    ax1.set_yscale("log")
    ax1.set_xlim(0, 1024)
    ax1.legend(loc="best", numpoints=1, prop={"size": 10})
    ax1.grid()

    plt.show()
