#
"""test_propagatedError.py
   Beispiel: Numerische Fehlerfortpflanzung mit PhyPraKit.prpagatedError()
   Illustriert auch die Verwendung der Rundung auf die Genauigkeit der
   Unsicherheit.

.. moduleauthor:: Guenter Quast <g.quast@kit.edu>
"""

from PhyPraKit import propagatedError, ustring
import numpy as np

# -example Code illustrating usage propagated Error and round_toError -----
if __name__ == "__main__":
    # Funktion von Messgrößen
    def func(a, b):
        return np.sqrt(a**2 + b**2)

    # Eingabewerte
    a = 1.12
    sa = 0.5
    b = 0.95
    sb = 0.5

    # Berechnung der Unsicherheit auf func(a,b)
    Delta_f = propagatedError(func, (a, b), (sa, sb))

    ## simple output
    print("\n", "*==* Numerical error propagation:\n", "    simple output:")
    print(
        "      func(",
        a,
        "+/-",
        sa,
        ",",
        b,
        "+/-",
        sb,
        ") : ",
        func(a, b),
        "+/-",
        Delta_f,
    )

    # output with percisions rounded to precision of uncertainty
    print("     correctly formatted output: ")
    print(
        "      func(",
        ustring(a, sa),
        ",",
        ustring(b, sb),
        ") = ",
        ustring(func(a, b), Delta_f),
    )
