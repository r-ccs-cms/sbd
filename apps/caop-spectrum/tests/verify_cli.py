"""Independent full Fock-space action and dense resolvent oracle; stdlib only."""
import csv, pathlib, sys
root = pathlib.Path(sys.argv[1])

def action(d, ops, fermion=True):
    sign = 1
    for q, addition in reversed(ops):
        if bool(d & (1 << q)) == addition:
            return None, 0
        if fermion:
            sign *= (-1) ** ((d & ((1 << q) - 1)).bit_count())
        d ^= 1 << q
    return d, sign

terms = [(v, [(i, True), (j, False)]) for i, j, v in
         [(0,0,.92), (1,1,.08), (0,1,-1.44), (1,0,-1.44),
          (2,2,.7), (3,3,1.3), (2,3,.2), (3,2,.2)]]
terms.append((.4, [(0,True), (2,True), (2,False), (0,False)]))

def solve(a, b):
    a = [list(x) + list(y) for x, y in zip(a,b)]
    n, p = len(a), len(b[0])
    for k in range(n):
        pivot = max(range(k,n),key=lambda i: abs(a[i][k]))
        a[k], a[pivot] = a[pivot], a[k]
        divisor = a[k][k]
        a[k] = [v/divisor for v in a[k]]
        for i in range(n):
            if i != k:
                v = a[i][k]
                a[i] = [x-v*y for x,y in zip(a[i],a[k])]
    return [r[n:] for r in a]

def expected(channel, w, eta, fermion=True):
    addition = channel == 'addition'
    basis = [d for d in range(16) if d.bit_count() == (2 if addition else 0)]
    h = [[0. for _ in basis] for _ in basis]
    for j,d in enumerate(basis):
        for v,ops in terms:
            target,s = action(d,ops,fermion)
            if target in basis:
                h[basis.index(target)][j] += v*s
    f = [[0. for _ in range(3)] for _ in basis]
    for d,v in [(1,.6),(2,.8)]:
        for q in range(3):
            target,s=action(d,[(q,addition)],fermion)
            if target in basis:
                f[basis.index(target)][q]+=s*v
    z = complex(w+(-1 if addition else 1),eta)
    a = [[(z if i==j else 0) + (-1 if addition else 1)*h[i][j]
          for j in range(len(basis))] for i in range(len(basis))]
    x = solve(a,f)
    g = [[sum(f[k][i]*x[k][j] for k in range(len(basis))) for j in range(3)] for i in range(3)]
    if not addition:
        g = list(zip(*g))
    return g

max_error = 0.
for path in root.glob('*.csv'):
    if path.name=='empty-h.csv':
        continue  # Separate single-term analytic oracle in test_cli.sh.
    channel = 'removal' if path.name.startswith('removal') else 'addition'
    eta = .21 if path.name=='new-eta.csv' else .13
    rows = list(csv.DictReader(path.open()))
    if path.name.startswith('zero'):
        assert len(rows)==201
        assert all(float(r[k])==0 for r in rows for k in ['G_real','G_imag','A_real','A_imag'])
        continue
    assert len(rows)==13*9, path
    for r in rows:
        i,j = int(r['orbital_a']),int(r['orbital_b'])
        g = expected(channel,float(r['omega']),eta,path.name!='boson.csv')
        value=complex(float(r['G_real']),float(r['G_imag']))
        max_error=max(max_error,abs(value-g[i][j])/(1+abs(g[i][j])))
        assert abs(value-g[i][j])<2e-9*(1+abs(g[i][j])), (path,r,g[i][j])
        import math
        spectral = -(g[i][j]-g[j][i].conjugate())/(2j*math.pi)
        actual=complex(float(r['A_real']),float(r['A_imag']))
        assert abs(actual-spectral)<2e-9*(1+abs(spectral)), (path,r,spectral)
assert (root/'replay.csv').read_bytes()==(root/'addition-1-1.csv').read_bytes()
print('PASS: independent Fock-space oracle, addition/removal, MPI 1/2/4, saved shards 1/2, extra basis, remap, coefficient replay and changed eta')
print(f'PASS: hard-core boson and all-zero seed; max scaled Green-function error {max_error:.3e}')
