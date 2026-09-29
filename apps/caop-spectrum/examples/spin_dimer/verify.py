"""Analytic singlet-to-triplet resolvent, including intersite signs."""
import csv, math, pathlib, sys
root=pathlib.Path(sys.argv[1]);maximum=0.
for filename,eta in [('spectrum.csv',.05),('broader.csv',.2)]:
    rows=list(csv.DictReader((root/filename).open()));assert len(rows)==101*16
    for row in rows:
        a,b=int(row['operator_a']),int(row['operator_b']);w=float(row['omega'])
        weight=(.25 if a==b else -.25) if a//2==b//2 else 0.
        expected=weight/complex(w-1,eta)
        actual=complex(float(row['G_real']),float(row['G_imag']))
        error=abs(actual-expected)/(1+abs(expected));maximum=max(maximum,error)
        assert error<1e-11,(filename,row,expected)
        spectral=-expected.imag/math.pi
        assert abs(complex(float(row['A_real']),float(row['A_imag']))-spectral)<1e-11*(1+abs(spectral))
print(f'PASS: spin dimer gap=1, weights +/-1/4, Sz/Sy cross terms=0, changed eta; scaled G error {maximum:.3e}')
