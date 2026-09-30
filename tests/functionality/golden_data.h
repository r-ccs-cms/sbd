/*
 * Golden reference data for test_rdm_opposite_spin_sign.cc.
 *
 * These are the 1-/2-RDM of an asymmetric H3 doublet (UHF-FCI) computed by an
 * INDEPENDENT oracle -- PySCF make_rdm12s -- and mapped into SBD storage layout.
 * No SBD source and no GPU are used to produce them, so the C++ test that checks
 * the kernel against these values is not circular.
 *
 * To regenerate this header, run the self-contained script below with numpy + pyscf
 * (any Python environment; no SBD build required):
 *
 *     python3 gen_golden.py > golden_data.h
 *
 * The values reproduce to the test tolerance (~1e-9); the last few digits vary with
 * SCF/FCI iterative convergence and are not significant.
 *
 * ======================== gen_golden.py ========================
 * #!/usr/bin/env python3
 * """Regenerate golden_data.h for test_rdm_opposite_spin_sign.cc.
 *
 * Self-contained: needs only numpy + pyscf. Produces the reference 1-/2-RDM from an
 * independent oracle (PySCF UHF-FCI make_rdm12s) and maps it into SBD storage layout.
 * No SBD source or GPU is involved. Run:  python3 gen_golden_standalone.py > golden_data.h
 *
 * SBD conventions reproduced:
 *   spin-orbital index = 2*spatial + spin (spin 0=alpha, 1=beta); det = interleaved bitmask.
 *   block b = si+2*sj at flat io + L*jo + L^2*ia + L^3*ja, block[io,jo,ia,ja] =
 *     <c^dag_{io,si} c^dag_{jo,sj} c_{ja,sj} c_{ia,si}>.
 * PySCF->SBD layout maps (verified element-wise):
 *   same-spin (aa): transpose (0,2,1,3) is self-inverse -> SBD = transpose(dm2aa,(0,2,1,3))
 *   opp-spin  (ab): pyscf = transpose(SBD,(1,3,0,2)) -> SBD = transpose(dm2ab,(2,0,3,1))
 *   1-RDM: dm1a[p,q]=<p^dag q> matches onebody[oi + L*oa] directly.
 * CI phase: PySCF orders (all alpha asc)(all beta asc); SBD interleaves. Fold the
 *   reordering Fermi sign (-1)^{#(p in aocc, q in bocc: p>q)} into each CI coefficient.
 * """
 * import numpy as np
 * from pyscf import gto, scf, fci, ao2mo
 * from pyscf.fci import cistring
 *
 * ATOM, BASIS, SPIN = "H 0 0 0; H 0 0 0.8; H 0 0 2.2", "sto-3g", 1  # H3_asym_doublet
 *
 * def reorder_sign(det, norb):
 *     aocc = [p for p in range(norb) if (det >> (2*p)) & 1]
 *     bocc = [q for q in range(norb) if (det >> (2*q+1)) & 1]
 *     inv = sum(1 for q in bocc for p in aocc if p > q)
 *     return -1.0 if (inv & 1) else 1.0
 *
 * mol = gto.M(atom=ATOM, basis=BASIS, spin=SPIN, verbose=0)
 * mf = scf.UHF(mol).run()
 * mo_a, mo_b = mf.mo_coeff
 * norb = mo_a.shape[1]
 * hcore = mf.get_hcore()
 * h1a = mo_a.T @ hcore @ mo_a; h1b = mo_b.T @ hcore @ mo_b
 * eri_aa = ao2mo.general(mol, (mo_a, mo_a, mo_a, mo_a), compact=False).reshape([norb]*4)
 * eri_ab = ao2mo.general(mol, (mo_a, mo_a, mo_b, mo_b), compact=False).reshape([norb]*4)
 * eri_bb = ao2mo.general(mol, (mo_b, mo_b, mo_b, mo_b), compact=False).reshape([norb]*4)
 * ecore = mol.energy_nuc(); nelec = mol.nelec
 * cis = fci.direct_uhf.FCI()
 * e_dav, ci = cis.kernel((h1a, h1b), (eri_aa, eri_ab, eri_bb), norb, nelec, ecore=ecore)
 * (dm1a, dm1b), (dm2aa, dm2ab, dm2bb) = cis.make_rdm12s(ci, norb, nelec)
 *
 * na, nb = nelec
 * strsa = cistring.gen_strings4orblist(range(norb), na)
 * strsb = cistring.gen_strings4orblist(range(norb), nb)
 * ci = ci.reshape(len(strsa), len(strsb))
 * dets, weights = [], []
 * for ia, sa in enumerate(strsa):
 *     for ib, sb in enumerate(strsb):
 *         d = 0
 *         for p in range(norb):
 *             if (sa >> p) & 1: d |= 1 << (2*p)
 *             if (sb >> p) & 1: d |= 1 << (2*p + 1)
 *         dets.append(d); weights.append(ci[ia, ib] * reorder_sign(d, norb))
 * weights = np.asarray(weights, dtype=float)
 * AB = np.transpose(dm2ab, (2,0,3,1)).reshape(-1)
 * AA = np.transpose(dm2aa, (0,2,1,3)).reshape(-1)
 * DM1A = dm1a.reshape(-1)
 *
 * def arr(name, vals):
 *     return "static const double %s[] = {%s};" % (name, ", ".join("%.17g" % v for v in vals))
 *
 * print("// AUTO-GENERATED golden data for test_rdm_opposite_spin_sign.cc")
 * print("// Source: PySCF make_rdm12s replica (test_rdm_vs_pyscf.py), case H3_asym_doublet")
 * print("// H 0 0 0; H 0 0 0.8; H 0 0 2.2 / sto-3g / spin=1 (UHF FCI). Do not edit by hand.")
 * print("#pragma once")
 * print("namespace golden {")
 * print("static const int NORB = %d;" % norb)
 * print("static const int NDET = %d;" % len(dets))
 * print("static const double E_DAV = %.17g;" % e_dav)
 * print("static const unsigned long DETS[] = {%s};" % ", ".join("0x%xUL" % d for d in dets))
 * print(arr("WEIGHTS", weights))
 * print(arr("AB", AB))
 * print(arr("AA", AA))
 * print(arr("DM1A", DM1A))
 * print("}")
 * ===============================================================
 */
#pragma once
namespace golden {
static const int NORB = 3;
static const int NDET = 9;
static const double E_DAV = -1.5884864228738571;
static const unsigned long DETS[] = {0x7UL, 0xdUL, 0x25UL, 0x13UL, 0x19UL, 0x31UL, 0x16UL, 0x1cUL, 0x34UL};
static const double WEIGHTS[] = {-0.9918303814299384, 6.9679885901591243e-05, -0.0099180322421520799, 0.0048443010504914397, -0.070950281505620952, 0.050177899915060192, -0.0097491293679435841, -0.031606169511278086, 0.086630799608516093};
static const double AB[] = {0.98375097278012491, 4.7227717638506048e-05, 0.0096694826996172999, -0.00027459389541857311, -0.00015310980016549494, -0.031347959161910237, -0.010080082555038105, -0.00041966567354844792, -0.085923059019295076, 4.7227717638506048e-05, 0.98382255105089, -0.0048047249586703755, -0.00069170347309031378, -0.00023902200757883445, 0.070370644768281532, -0.00048919083768364719, -0.010681580574399944, 0.049767965612107425, 0.0096694826996172999, -0.0048047249586703755, 0.00011851277610069253, 6.7931822199816134e-07, -3.3755034447120213e-07, -0.00065183715862094727, -9.6692179404176196e-05, 4.8045934009465288e-05, -0.0010876517259018003, -0.00027459389541857311, -0.00069170347309031378, 6.7931822199816134e-07, 0.0050339473010133579, 0.0022424666241395545, -2.202314285332209e-06, 0.0035594450369793927, 0.0061464796192812546, -6.0364242322850166e-06, -0.00015310980016549494, -0.00023902200757883445, -3.3755034447120213e-07, 0.0022424666241395545, 0.00099895480646214362, 4.9438075199974472e-06, 0.0015859312104353387, 0.0027373766499693216, 3.4963903408628591e-06, -0.031347959161910237, 0.070370644768281532, -0.00065183715862094727, -2.202314285332209e-06, 4.9438075199974472e-06, 0.0060328923969025028, 0.00031347100826378008, -0.00070368717956251496, 0.0062982038616587171, -0.010080082555038105, -0.00048919083768364719, -9.6692179404176196e-05, 0.0035594450369793927, 0.0015859312104353387, 0.00031347100826378008, 0.0026161890034421659, 0.0043469515923177564, 0.00085920706368067842, -0.00041966567354844792, -0.010681580574399944, 4.8045934009465288e-05, 0.0061464796192812546, 0.0027373766499693216, -0.00070368717956251496, 0.0043469515923177564, 0.0076032628043672407, -0.00049766602920104709, -0.085923059019295076, 0.049767965612107425, -0.0010876517259018003, -6.0364242322850166e-06, 3.4963903408628591e-06, 0.0062982038616587171, 0.00085920706368067842, -0.00049766602920104709, 0.01002271708069667};
static const double AA[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.98382587774630004, -0.005297447180351425, -0.98382587774630004, 0, -0.010526487449012646, 0.005297447180351425, 0.010526487449012646, 0, 0, -0.005297447180351425, 0.0075752313382804488, 0.005297447180351425, 0, 0.0066366459340958164, -0.007575231338280434, -0.0066366459340958164, 0, 0, -0.98382587774630004, 0.005297447180351425, 0.98382587774630004, 0, 0.010526487449012646, -0.005297447180351425, -0.010526487449012646, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.010526487449012646, 0.0066366459340958164, 0.010526487449012646, 0, 0.008598890915419417, -0.0066366459340958164, -0.0085988909154194257, 0, 0, 0.005297447180351425, -0.007575231338280434, -0.005297447180351425, 0, -0.0066366459340958164, 0.0075752313382804488, 0.0066366459340958164, 0, 0, 0.010526487449012646, -0.0066366459340958164, -0.010526487449012646, 0, -0.0085988909154194257, 0.0066366459340958164, 0.008598890915419417, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static const double DM1A[] = {0.99140110908458046, 0.0066366459340958164, 0.010526487449012646, 0.0066366459340958164, 0.99242476866171947, -0.005297447180351425, 0.010526487449012646, -0.005297447180351425, 0.016174122253699866};
}
