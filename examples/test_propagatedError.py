"""test_propagatedError.py
   Beispiel: Numerische Fehlerfortpflanzung mit PhyPraKit.prpagatedError()
   Illustriert auch die Verwendung der Rundung auf die Genauigkeit der
   Unsicherheit.
   
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

# format specifier for v+/-e with pv and pe significant digigs
nd_e = 2 # set number of significant digits for uncertainty
fmttxt="{:#.{pv}g}+/-{:#.{pe}g}"
#    determine number of significant digits for a, b and f
nd_a, _va, _sa = ppk.round_to_error(a, sa, nd_e)
nd_b, _vb, _sb = ppk.round_to_error(b, sb, nd_e)
nd_f, _vf, _sf = ppk.round_to_error(func(a,b), Delta_f, nd_e)
#    format output with proper number of significant digits ...
txt_a = fmttxt.format(_va, _sa, pv=nd_a, pe=nd_e)
txt_b = fmttxt.format(_vb, _sb, pv=nd_b, pe=nd_e)
txt_f = fmttxt.format(_vf, _sf, pv=nd_f, pe=nd_e)
#    ... print
print(txt_a, ',', txt_b,' : ', txt_f)

## simple output
##print('func(', a, '+/-', sa, ',', b, '+/-', sb, ') : ',
##       func(a,b), '+/-', Delta_f)

