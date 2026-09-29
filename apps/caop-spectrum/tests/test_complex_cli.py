"""Complex CAOP CLI versus independent full Fock-space action/resolvents."""
import csv, math, os, pathlib, subprocess, sys, tempfile
from mpi_command import mpi_command
exe=os.environ.get('CAOP_SPECTRUM_EXE','./caop-spectrum')
root=pathlib.Path(sys.argv[1]);work=pathlib.Path(tempfile.mkdtemp(prefix='complex-',dir=root))
def number(c):
    c=complex(c);return f'({c.real:.17g},{c.imag:.17g})'
def write_op(path,terms,fermion=True):
    path.write_text(('-1' if fermion else '+1')+'\n'+''.join(number(c)+''.join(f' {"cdag" if add else "c"} {q}' for q,add in seq)+'\n' for c,seq in terms))
def action(d,seq,fermion):
    sign=1
    for q,add in reversed(seq):
        if bool(d&(1<<q))==bool(add):return None,0
        if fermion:sign*=(-1)**((d&((1<<q)-1)).bit_count())
        d^=1<<q
    return d,sign
hterms=[(v,[(i,1),(j,0)]) for i,j,v in [(0,0,.92),(1,1,.08),(0,1,1.44j),(1,0,-1.44j),(2,2,.7),(3,3,1.3),(2,3,.2+.35j),(3,2,.2-.35j)]]+[(.4,[(0,1),(2,1),(2,0),(0,0)])]
ops=[[(.3+.2j,[]),(.7,[(0,1),(0,0)]),(1.2j,[(0,1),(1,0)]),(.4-.3j,[(2,1)]),(-.6j,[(1,0)])],
     [(-.2j,[]),(.8-.5j,[(1,1),(0,0)]),(.9j,[(3,1),(2,1)]),(.45,[(0,0)])],
     [(0.,[])],[(1.,[(0,0)]),(.75j,[(1,0)])],[(.5j,[(2,1)])]]
(work/'basis.txt').write_text(''.join(f'{d:04b}\n' for d in range(16)))
(work/'truncated.txt').write_text('0010\n')
for fermion in [True,False]:
    tag='fermion' if fermion else 'boson';write_op(work/f'{tag}-ham.txt',hterms,fermion)
    for v,terms in enumerate(ops):write_op(work/f'{tag}-op{v}.txt',terms,fermion)
# SBD spin convention: s+ annihilates, s- creates. This file exercises its Sy parser.
(work/'sy.txt').write_text('+1\n1 sy 2\n')

def expected(w,eta,operators,psi,fermion,removal=False):
    n,p=16,len(operators);h=[[0j]*n for _ in range(n)];f=[[0j]*p for _ in range(n)]
    for d in range(n):
        for c,seq in hterms:
            t,s=action(d,seq,fermion)
            if t is not None:h[t][d]+=c*s
    for v,terms in enumerate(operators):
        for d,amplitude in psi:
            for c,seq in terms:
                t,s=action(d,seq,fermion)
                if t is not None:f[t][v]+=c*s*amplitude
    z=complex(w+(1 if removal else -1),eta)
    a=[[(z if i==j else 0)+(1 if removal else -1)*h[i][j] for j in range(n)]+f[i][:] for i in range(n)]
    for k in range(n):
        pivot=max(range(k,n),key=lambda i:abs(a[i][k]));a[k],a[pivot]=a[pivot],a[k]
        c=a[k][k];a[k]=[x/c for x in a[k]]
        for i in range(n):
            if i!=k:
                c=a[i][k];a[i]=[x-c*y for x,y in zip(a[i],a[k])]
    g=[[sum(f[k][i].conjugate()*a[k][n+j] for k in range(n)) for j in range(p)] for i in range(p)]
    return list(zip(*g)) if removal else g

def run(args,label,threads=2,success=True):
    with (work/f'{label}.log').open('w') as out:
        result=subprocess.run(args,timeout=60,stdout=out,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS=str(threads)))
    assert (result.returncode==0)==success,(label,work)
maximum=0.
def check(path,operators,psi,fermion,removal=False,eta=.17):
    global maximum
    rows=list(csv.DictReader(path.open()));assert len(rows)==7*len(operators)**2
    for row in rows:
        i=int(row.get('operator_a',row.get('orbital_a')));j=int(row.get('operator_b',row.get('orbital_b')))
        g=expected(float(row['omega']),eta,operators,psi,fermion,removal)
        got=complex(float(row['G_real']),float(row['G_imag']));error=abs(got-g[i][j])/(1+abs(g[i][j]));maximum=max(maximum,error)
        assert error<3e-9,(path,row,g[i][j])
        spectral=-(g[i][j]-g[j][i].conjugate())/(2j*math.pi)
        assert abs(complex(float(row['A_real']),float(row['A_imag']))-spectral)<3e-9*(1+abs(spectral))

def case(label,operators,files=None,channel=None,fermion=True,ranks=4,hs=1,ts=1,shards=1,real_input=False,threads=2,normalize=False,hamiltonian_mode="stored"):
    tag='fermion' if fermion else 'boson';prefix=('state-' if real_input else ('complex-state-' if shards==1 else 'complex-two-'))
    coeff=work/f'{label}.coeff';output=work/f'{label}.csv'
    args=mpi_command(ranks,exe)+['--scalar-type','complex','--hamfile',str(work/f'{tag}-ham.txt'),'--loadname',str(root/prefix),'--wavefunction-shards',str(shards),'--sites','4','--reference-energy','-1','--extra-detfile',str(work/'basis.txt'),'--remap-detfile',str(work/'truncated.txt') if normalize else str(root/'remap.txt'),'--save-coefficients',str(coeff),'--output',str(output),'--steps','16','--omega-min','-2','--omega-max','3','--points','7','--eta','.17','--h-comm-size',str(hs),'--t-comm-size',str(ts)]
    args += ["--hamiltonian-mode",hamiltonian_mode]
    if normalize:args+=['--normalize-remap']
    if real_input:args+=['--wavefunction-type','real']
    if channel:args+=['--channel',channel,'--orbitals','0,1,2']
    else:
        args+=['--applied-operator-type','general']
        for file in files:args+=['--operator-file',str(file)]
    run(args,label,threads)
    psi=[(2,1j)] if normalize else [(1,.6),(2,.8 if real_input else .8j)]
    check(output,operators,psi,fermion,channel=='removal')
    replay=work/f'{label}-replay.csv'
    base=[exe,'--read-coefficients',str(coeff),'--output',str(replay),'--omega-min','-2','--omega-max','3','--points','7']
    run(base+['--eta','.17'],label+'-replay',threads)
    assert replay.read_bytes()==output.read_bytes()
    run(base+['--eta','.23'],label+'-eta',threads)
    check(replay,operators,psi,fermion,channel=='removal',.23)
    if normalize:
        import re
        log=(work/f'{label}.log').read_text()
        for key,value in [('retained_norm2',.64),('discarded_norm2',.36)]:assert abs(float(re.search(key+r'=([^\s]+)',log).group(1))-value)<1e-14
    return args

backend='stored'
for ranks,hs,ts,shards,threads in [(1,1,1,2,1),(2,1,1,1,2),(4,1,1,2,3),(4,2,1,1,2),(4,1,2,2,3),(4,2,2,1,2)]:
    label=f'{backend}-{ranks}-{hs}-{ts}'
    args=case(label,ops,files=[work/f'fermion-op{v}.txt' for v in range(len(ops))],ranks=ranks,hs=hs,ts=ts,shards=shards,threads=threads)
for channel in ['addition','removal']:
    case(backend+'-'+channel,[[(1,[(v,channel=='addition')])] for v in range(3)],channel=channel,hs=2,ts=2)
case(backend+'-real-input',ops,files=[work/f'fermion-op{v}.txt' for v in range(len(ops))],real_input=True,hs=2)
case(backend+'-normalized',ops,files=[work/f'fermion-op{v}.txt' for v in range(len(ops))],normalize=True,ts=2)
case(backend+'-sy',[[(.5j,[(2,1)]),(-.5j,[(2,0)])]],files=[work/'sy.txt'],fermion=False,hs=2,ts=2)
case(backend+'-zero',[ops[2]],files=[work/'fermion-op2.txt'])
for ranks,hs,ts,threads in [(1,1,1,1),(2,1,1,2),(4,1,1,3),(4,2,1,2),(4,1,2,3),(4,2,2,2)]:
    case(f'on-the-fly-{ranks}-{hs}-{ts}',ops,files=[work/f'fermion-op{v}.txt' for v in range(len(ops))],ranks=ranks,hs=hs,ts=ts,threads=threads,hamiltonian_mode='on-the-fly')
for channel in ['addition','removal']:
    case('on-the-fly-'+channel,[[(1,[(v,channel=='addition')])] for v in range(3)],channel=channel,hs=2,ts=2,hamiltonian_mode='on-the-fly')
case('on-the-fly-sy',[[(.5j,[(2,1)]),(-.5j,[(2,0)])]],files=[work/'sy.txt'],fermion=False,hs=2,ts=2,hamiltonian_mode='on-the-fly')
run([exe,'--hamiltonian-mode','bad'],'reject-hamiltonian-mode',success=False)
run(args+['--wavefunction-type','real'],'reject-checkpoint-type',success=False)
print(f'PASS: complex H/state/operators, Sy, conjugate response/removal, checkpoint shards/promotion, remap norms, stored/on-the-fly MPI/h/b/t OpenMP; scaled G error {maximum:.3e}; logs {work}')
