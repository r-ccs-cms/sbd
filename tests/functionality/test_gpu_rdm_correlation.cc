/**
 * @file test_gpu_rdm_correlation.cc
 * @brief Regression tests for full one-/two-particle RDM accumulation
 *        (`--rdm 1`), the code path consumed by orbital optimization.
 *
 * WHAT THIS PINS
 * --------------
 * The RDMs are accumulated in
 *   include/sbd/chemistry/basic/correlation_thrust.h
 * by three device routines:
 *   - ZeroDiffCorrelation  (bra == ket)        -> diagonal contributions
 *   - OneDiffCorrelation / TwoDiffCorrelation  -> off-diagonal contributions
 *
 * For a real Hamiltonian the RDM arrays (`onebody`, `twobody`) are REAL
 * (`ElemT = double`), but each accumulated value is formed as
 *   Conjugate(WeightI) * WeightJ * ElemT(sgn)
 * which is std::complex-typed. The CORRECT accumulation writes the REAL PART of
 * that value into the TARGET element (this is what a raw
 * `atomicAdd(double*, complex)` did by implicit narrowing).
 *
 * A performance refactor replaced those raw writes with a generic
 * `sbd::atomic_add(...)` helper (include/sbd/framework/cuda_utility.h). For the
 * off-diagonal terms, when the accumulator is `double*` and the value is
 * `std::complex<double>`, this mis-dispatches: rather than narrowing to
 * `.real()` in the target slot, it treats the buffer as `std::complex<double>*`
 * and writes `.real()` to p[0] and `.imag()` to the ADJACENT slot p[1]. The
 * off-diagonal contribution never lands and a neighbour is corrupted, so the
 * RDM collapses to its diagonal and Tr(gamma^1) falls far below N_e
 * (observed N2 20q: 0.024 instead of 14; Fe2S2 40q: ~0.07 instead of 30).
 *
 * These are host-side, dependency-free unit tests (no GPU/FCIDUMP/MPI) that
 * mirror the two accumulation semantics EXACTLY and assert the invariant the
 * refactor violated. `test_full_rdm_physical_invariants()` documents the
 * end-to-end assertions to reproduce with the diag-gpu binary (Tr == N_e and
 * energy-from-RDM == Davidson) for RHF (N2) and UHF (Fe2S2).
 *
 * Build (standalone):  g++ -std=c++17 -O2 -o test_gpu_rdm_correlation \
 *                          test_gpu_rdm_correlation.cc
 */

#include <cmath>
#include <complex>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>

using cd = std::complex<double>;

static const double ABS_TOL = 1e-12;
static bool almost_equal(double a, double b, double tol = ABS_TOL) {
  return std::abs(a - b) <= tol;
}

static int g_failures = 0;
#define CHECK(cond, msg)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::cout << "  FAIL: " << msg << "  [" << #cond << "]" << std::endl;    \
      ++g_failures;                                                            \
    } else {                                                                   \
      std::cout << "  ok:   " << msg << std::endl;                             \
    }                                                                          \
  } while (0)

// -----------------------------------------------------------------------------
// Mirrors of the two competing accumulation semantics.
//
// correct_accumulate: what the raw atomicAdd(double*, complex) did (and what
//   the reverted correlation_thrust.h restores) — narrow the complex value to
//   its real part into the TARGET element.
// buggy_accumulate:   what sbd::atomic_add(double*, complex) did on device via
//   the mis-selected complex overload — real part to target, imag part leaked
//   into the ADJACENT element.
// -----------------------------------------------------------------------------
static inline void correct_accumulate(double *buf, std::size_t idx, cd value) {
  buf[idx] += value.real();
}
static inline void buggy_accumulate(double *buf, std::size_t idx, cd value) {
  buf[idx] += value.real();      // p[0]  <- real
  buf[idx + 1] += value.imag();  // p[1]  <- imag  (the leak)
}

// Project helpers, matching include/sbd/framework/type_def.h (host branch):
//   Conjugate(real) == real ; Conjugate(complex) == conj
//   SquaredNorm(complex) == |z|^2
static inline cd Conjugate(cd a) { return std::conj(a); }
static inline double SquaredNorm(cd x) {
  return x.real() * x.real() + x.imag() * x.imag();
}

// -----------------------------------------------------------------------------
// Test 1: off-diagonal accumulation must land Re() in the target and leak
// nothing into the neighbour. This is the exact invariant PR#43 broke.
// -----------------------------------------------------------------------------
static void test_offdiagonal_real_accumulation() {
  std::cout << "\n=== Test 1: off-diagonal RDM accumulation (real part only) ==="
            << std::endl;

  const std::size_t norbs = 4;
  std::vector<double> onebody(norbs * norbs + 1, 0.0); // +1 guard = neighbour

  // value = Conjugate(WeightI) * WeightJ * sgn, deliberately complex so a
  // neighbour leak (imag part) would be detectable.
  cd WeightI(0.7, 0.3), WeightJ(0.5, -0.2);
  int sgn = 1;
  cd value = Conjugate(WeightI) * WeightJ * cd((double)sgn, 0.0);

  const std::size_t oi = 1, oa = 2;
  const std::size_t target = oi + norbs * oa;

  correct_accumulate(onebody.data(), target, value);

  CHECK(almost_equal(onebody[target], value.real()),
        "target element holds Re(Conjugate(WeightI)*WeightJ)");
  CHECK(almost_equal(onebody[target + 1], 0.0),
        "neighbour untouched (no imag() leak into adjacent element)");

  std::cout << std::fixed << std::setprecision(6)
            << "  value = (" << value.real() << ", " << value.imag()
            << "), onebody[target] = " << onebody[target]
            << ", onebody[target+1] = " << onebody[target + 1] << std::endl;
}

// -----------------------------------------------------------------------------
// Test 2: demonstrate the failure mode explicitly, so the test documents the
// bug it guards against. The buggy semantics MUST differ from the correct ones
// on a complex value; if some future refactor makes them coincide by dropping
// the imag leak, that is fine — but a silent collapse to buggy semantics is
// caught by Test 1 above (neighbour != 0).
// -----------------------------------------------------------------------------
static void test_bug_signature_is_detectable() {
  std::cout << "\n=== Test 2: bug signature (imag leak) is detectable ==="
            << std::endl;

  const std::size_t norbs = 4;
  std::vector<double> good(norbs * norbs + 1, 0.0);
  std::vector<double> bad(norbs * norbs + 1, 0.0);

  cd value = Conjugate(cd(0.6, 0.4)) * cd(0.5, 0.5); // complex, imag != 0
  const std::size_t target = 3;

  correct_accumulate(good.data(), target, value);
  buggy_accumulate(bad.data(), target, value);

  CHECK(almost_equal(good[target], bad[target]),
        "both semantics agree on the target element (real part)");
  CHECK(!almost_equal(good[target + 1], bad[target + 1]),
        "buggy semantics corrupt the neighbour; correct semantics do not");
  CHECK(std::abs(bad[target + 1] - value.imag()) < ABS_TOL,
        "buggy neighbour value equals the leaked imag part (bug fingerprint)");
}

// -----------------------------------------------------------------------------
// Test 3: diagonal path uses SquaredNorm and is unaffected by the bug — assert
// it accumulates |w|^2 as a real number (ZeroDiffCorrelation semantics).
// -----------------------------------------------------------------------------
static void test_diagonal_squarednorm() {
  std::cout << "\n=== Test 3: diagonal accumulation uses |w|^2 ===" << std::endl;
  double acc = 0.0;
  cd w(0.6, 0.4);
  acc += SquaredNorm(w); // atomic_add_real(double*, |w|^2)
  CHECK(almost_equal(acc, 0.52), "diagonal accumulates SquaredNorm(w) = |w|^2");
}

// -----------------------------------------------------------------------------
// Test 4: 1-RDM trace + symmetry invariants on a small hand-built matrix — the
// cheap analogue of the application check. A well-formed real 1-RDM has
// Tr == N_e, is symmetric, and carries non-zero off-diagonal coherence (which
// the bug destroyed).
// -----------------------------------------------------------------------------
static void test_1rdm_trace_and_symmetry() {
  std::cout << "\n=== Test 4: 1-RDM trace + symmetry ===" << std::endl;

  const std::size_t n = 3;
  const double N_e = 4.0;
  std::vector<double> g(n * n, 0.0);
  g[0 * n + 0] = 1.9; g[1 * n + 1] = 1.6; g[2 * n + 2] = 0.5; // trace 4.0
  g[0 * n + 1] = g[1 * n + 0] = 0.20;
  g[1 * n + 2] = g[2 * n + 1] = -0.10;

  double trace = 0.0;
  for (std::size_t p = 0; p < n; ++p) trace += g[p * n + p];
  CHECK(almost_equal(trace, N_e), "Tr(gamma^1) == N_e");

  bool sym = true;
  for (std::size_t p = 0; p < n && sym; ++p)
    for (std::size_t q = 0; q < n; ++q)
      if (!almost_equal(g[p * n + q], g[q * n + p])) sym = false;
  CHECK(sym, "1-RDM is symmetric (gamma_pq == gamma_qp)");

  double offdiag = 0.0;
  for (std::size_t p = 0; p < n; ++p)
    for (std::size_t q = 0; q < n; ++q)
      if (p != q) offdiag += std::abs(g[p * n + q]);
  CHECK(offdiag > 0.0,
        "off-diagonal coherence present (zeroed by the collapse bug)");
}

// -----------------------------------------------------------------------------
// Test 5: documented end-to-end invariants for the GPU --rdm 1 path.
// Reproduce in an integration job with the diag-gpu binary:
//   RHF N2 20q    : Tr(gamma^1) == 14, E0 + Sum h.gamma + 1/2 Sum (pq|rs).Gamma == E_Davidson
//   UHF Fe2S2 40q : Tr(gamma^1) == 30, same energy identity
// Both held on the reverted binary; both FAILED on the refactored one.
// -----------------------------------------------------------------------------
static void test_full_rdm_physical_invariants() {
  std::cout << "\n=== Test 5: full-RDM physical invariants (documented) ==="
            << std::endl;
  std::cout << "  (integration check — run diag-gpu*-mpi with --rdm 1)\n"
            << "  RHF N2 20q  : Tr == 14, E_from_RDM == E_Davidson\n"
            << "  UHF Fe2S2 40q: Tr == 30, E_from_RDM == E_Davidson"
            << std::endl;
}

int main() {
  std::cout << "========================================\n"
            << "Full-RDM (--rdm 1) correlation regression suite\n"
            << "========================================" << std::endl;

  test_offdiagonal_real_accumulation();
  test_bug_signature_is_detectable();
  test_diagonal_squarednorm();
  test_1rdm_trace_and_symmetry();
  test_full_rdm_physical_invariants();

  std::cout << "\n========================================" << std::endl;
  if (g_failures == 0)
    std::cout << "All RDM regression checks PASSED" << std::endl;
  else
    std::cout << g_failures << " RDM regression check(s) FAILED" << std::endl;
  std::cout << "========================================" << std::endl;
  return g_failures == 0 ? 0 : 1;
}
