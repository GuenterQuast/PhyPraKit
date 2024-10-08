"""test_readCassy.py
   read data exported by Leybold Cassylab in .txt format

.. moduleauthor:: Guenter Quast <g.quast@kit.edu>

"""

# -----example Code illustrating usage --------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from PhyPraKit import readCassy
    import sys

    # check for / read command line arguments
    if len(sys.argv) == 2:
        fname = sys.argv[1]
    else:
        fname = "Cassy.txt"
    print(
        (
            "\n*==* script " + sys.argv[0] + " executing \n",
            "     processing file " + fname,
        )
    )

    # read data exported by CASSY in .txt format
    tags, data = readCassy(fname, prlevel=2)
    ic = len(data)

    t = data[0]
    # make a plot
    fig = plt.figure(1, figsize=(10.0, 5.0))
    ax1 = fig.add_subplot(1, 1, 1)
    l = min(len(t), 5000)
    for i in range(1, ic):
        ax1.plot(t[:l], data[i, :l])

    plt.show()
