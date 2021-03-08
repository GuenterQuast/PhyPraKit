"""test_propagatedError.py
   Beispiel: Numerische Fehlerfortpflanzung mit PhyPraKit.prpagatedError()
   
.. moduleauthor:: Guenter Quast <g.quast@kit.edu>
"""
import PhyPraKit as ppk

# Funktion von Messgrößen
def func(a, b):
    return a + b

# Eingabewerte
a=1. ; sa=0.5
b=1. ; sb=0.5

# Berechnung der Unsicherheit auf func(a,b)
Delta_f = ppk.propagatedError(func, (a, b), (sa, sb) )

# Ergebnisausgabe
print('func(', a, '+/-', sa, ',', b, '+/-', sb, ') : ',
       func(a,b), '+/-', Delta_f)

