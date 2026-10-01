"""CLI integration with an independent 16-state Fock resolvent (stdlib only)."""
import csv, math, os, pathlib, subprocess, sys, tempfile
from mpi_command import mpi_command
exe=os.environ.get('CAOP_SPECTRUM_EXE','./caop-spectrum')
root=pathlib.Path(sys.argv[1]); work=pathlib.Path(tempfile.mkdtemp(prefix='general-',dir=root))
# Non-number-conserving, non-Hermitian operators; diagonal/offdiagonal
# contributions reach the same target across different parent ranks.
ops=[[(.3,[]),(.7,[(0,1),(0,0)]),(1.2,[(0,1),(1,0)]),
      (.4,[(2,1)]),(-.6,[(1,0)])],
     [(-.2,[]),(.8,[(1,1),(0,0)]),(.9,[(3,1),(2,1)]),(.45,[(0,0)])],
     [(0.,[])],[(1.,[(0,0)]),(-.75,[(1,0)])],[(.5,[(2,1)])]]
for v,terms in enumerate(ops):
    (work/f'op{v}.txt').write_text('-1\n'+''.join(str(c)+''.join(f' {"cdag" if add else "c"} {q}' for q,add in seq)+'\n' for c,seq in terms))
(work/'basis.txt').write_text(''.join(f'{d:04b}\n' for d in range(16)))
def action(d,seq):
    sign=1
    for q,add in reversed(seq):
        if bool(d&(1<<q))==bool(add):return None,0
        sign*=(-1)**((d&((1<<q)-1)).bit_count());d^=1<<q
    return d,sign
hterms=[(v,[(i,1),(j,0)]) for i,j,v in [(0,0,.92),(1,1,.08),(0,1,-1.44),(1,0,-1.44),(2,2,.7),(3,3,1.3),(2,3,.2),(3,2,.2)]]+[(.4,[(0,1),(2,1),(2,0),(0,0)])]
h=[[0.]*16 for _ in range(16)];f=[[0.]*len(ops) for _ in range(16)]
for d in range(16):
    for c,seq in hterms:
        t,s=action(d,seq)
        if t is not None:h[t][d]+=c*s
for v,terms in enumerate(ops):
    for d,amplitude in [(1,.6),(2,.8)]:
        for c,seq in terms:
            t,s=action(d,seq)
            if t is not None:f[t][v]+=c*s*amplitude

def expected(w):
    n,p=16,len(ops);z=complex(w-1,.17)
    a=[[(z if i==j else 0)-h[i][j] for j in range(n)]+f[i][:] for i in range(n)]
    for k in range(n):
        pivot=max(range(k,n),key=lambda i:abs(a[i][k]));a[k],a[pivot]=a[pivot],a[k]
        c=a[k][k];a[k]=[x/c for x in a[k]]
        for i in range(n):
            if i!=k:
                c=a[i][k];a[i]=[x-c*y for x,y in zip(a[i],a[k])]
    return [[sum(f[k][i]*a[k][n+j] for k in range(n)) for j in range(p)] for i in range(p)]

def run(args,log,success=True):
    with log.open('w') as out:
        result=subprocess.run(args,timeout=60,stdout=out,stderr=subprocess.STDOUT)
    assert (result.returncode==0)==success,log
maximum=0
for backend,ranks,hs,ts in [(mode,*geometry) for mode in ('stored','on-the-fly')
                            for geometry in [(1,1,1),(2,1,1),(4,1,1),(4,2,1),(4,1,2),(4,2,2)]]:
    label=f'{backend}-{ranks}-{hs}-{ts}';output=work/f'{label}.csv';coeff=work/f'{label}.coeff'
    args=mpi_command(ranks,exe)+['--hamfile',str(root/'ham.txt'),'--loadname',str(root/'state-'),'--wavefunction-shards','1','--sites','4','--reference-energy','-1','--applied-operator-type','general','--extra-detfile',str(work/'basis.txt'),'--remap-detfile',str(root/'remap.txt'),'--save-coefficients',str(coeff),'--output',str(output),'--steps','16','--omega-min','-2','--omega-max','3','--points','7','--eta','.17','--h-comm-size',str(hs),'--t-comm-size',str(ts)]
    args+=['--hamiltonian-mode',backend]
    for v in range(len(ops)):args+=['--operator-file',str(work/f'op{v}.txt')]
    run(args,work/f'{label}.log')
    rows=list(csv.DictReader(output.open()));assert len(rows)==7*len(ops)**2
    for row in rows:
        i,j=int(row['operator_a']),int(row['operator_b']);g=expected(float(row['omega']))
        got=complex(float(row['G_real']),float(row['G_imag']));err=abs(got-g[i][j])/(1+abs(g[i][j]));maximum=max(maximum,err);assert err<2e-9,(label,row,g[i][j])
        spectral=-(g[i][j]-g[j][i].conjugate())/(2j*math.pi)
        assert abs(complex(float(row['A_real']),float(row['A_imag']))-spectral)<2e-9*(1+abs(spectral))
    replay=work/f'{label}-replay.csv'
    run([exe,'--read-coefficients',str(coeff),'--output',str(replay),'--omega-min','-2','--omega-max','3','--points','7','--eta','.17'],work/f'{label}-replay.log')
    assert replay.read_bytes()==output.read_bytes()
# Rejected inputs must fail before attempting inconsistent collective work.
(work/'boson.txt').write_text('+1\n1 cdag 0\n')
for label,extra in [('statistics',['--operator-file',str(work/'boson.txt')]),('channel',['--channel','addition']),('range',['--operator-file',str(work/'bad.txt')])]:
    (work/'bad.txt').write_text('-1\n1 cdag 4\n')
    run(args+extra,work/f'reject-{label}.log',False)
# Entirely zero operator exercises empty-chain general-operator save/replay.
zero_args=[]
it=iter(args)
for arg in it:
    if arg=='--operator-file':next(it)
    else:zero_args.append(arg)
zero_args+=['--operator-file',str(work/'op2.txt')]
run(zero_args,work/'zero.log')
for row in csv.DictReader(output.open()):
    assert all(float(row[k])==0 for k in ['G_real','G_imag','A_real','A_imag'])
run([exe,'--read-coefficients',str(coeff),'--output',str(replay),'--omega-min','-2','--omega-max','3','--points','7','--eta','.17'],work/'zero-replay.log')
assert replay.read_bytes()==output.read_bytes()
print(f'PASS: general CAOP operators, duplicate sums/cancellation, zero column, mixed sectors, h/b/t, remap/replay; scaled G error {maximum:.3e}')
