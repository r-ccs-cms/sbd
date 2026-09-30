/**
 * @file test_rdm_opposite_spin_sign.cc
 * @brief Regression test for the opposite-spin (ab/ba) 2-RDM double-excitation
 *        sign fix in TwoDiffCorrelation.
 *
 * The 2-RDM is built directly from a genuine full-CI vector (UHF FCI of an
 * asymmetric H3 doublet, produced by PySCF) by scattering every ordered
 * determinant pair through the production correlation kernel
 * (sbd/chemistry/basic/correlation.h::CorrelationTermAddition, which dispatches
 * to Zero/One/TwoDiffCorrelation). The resulting spin-resolved blocks are then
 * compared element-wise to golden values dumped from PySCF make_rdm12s
 * (tests/functionality/golden_data.h -- see that header for provenance).
 *
 * Discriminator: the opposite-spin ab block (twobody[2]). The original
 * sorted-index pairing (min(i,j)->min(a,b)) with two bra parities pairs
 * operators across spins and ignores the intermediate determinant, so this
 * block is wrong by ~5e-2 and the test FAILS. With the sign fix (spin-consistent
 * pairing + intermediate determinant) it matches PySCF to ~1e-16 and PASSES.
 *
 * Controls: the same-spin aa block (twobody[0]) and the alpha 1-RDM
 * (onebody[0]) are unchanged by the fix and must match PySCF either way; they
 * guard against a regression in the untouched branches.
 *
 * Kernel path: this drives the host free functions in
 * sbd/chemistry/basic/correlation.h. The GPU/tpb device source
 * sbd/chemistry/basic/correlation_thrust.h::CorrelationKernels::TwoDiffCorrelation
 * carries byte-identical opposite-spin sign arithmetic (the fix edits both in
 * lockstep); its production path is additionally verified on the GPU binary
 * (diag-gpu_uhf_diag, NORB=20: E_recon-E_davidson +0.937 mHa -> -6.25e-13 Ha).
 */
#include "sbd/chemistry/basic/determinants.h"
#include "sbd/chemistry/basic/correlation.h"
#include "utils.h"
#include "golden_data.h"
#include <mpi.h>
#include <vector>
#include <cmath>

using namespace sbd;

namespace {
const size_t BIT = 64;

void build_rdm(std::vector<std::vector<double>> &onebody,
               std::vector<std::vector<double>> &twobody) {
  const size_t L = golden::NORB;
  onebody.assign(2, std::vector<double>(L * L, 0.0));
  twobody.assign(4, std::vector<double>(L * L * L * L, 0.0));
  std::vector<int> c(4, 0), d(4, 0);
  for (int I = 0; I < golden::NDET; ++I) {
    for (int J = 0; J < golden::NDET; ++J) {
      std::vector<size_t> DI = {static_cast<size_t>(golden::DETS[I])};
      std::vector<size_t> DJ = {static_cast<size_t>(golden::DETS[J])};
      CorrelationTermAddition<double>(DI, DJ, golden::WEIGHTS[I],
                                      golden::WEIGHTS[J], BIT, L, c, d,
                                      onebody, twobody);
    }
  }
}

double max_abs_diff(const std::vector<double> &a, const double *g, size_t n) {
  double m = 0.0;
  for (size_t i = 0; i < n; ++i)
    m = std::max(m, std::abs(a[i] - g[i]));
  return m;
}
} // namespace

void test_opposite_spin_ab_block() {
  std::vector<std::vector<double>> onebody, twobody;
  build_rdm(onebody, twobody);
  const size_t L = golden::NORB;
  const size_t n4 = L * L * L * L;
  const double dab = max_abs_diff(twobody[2], golden::AB, n4);
  TEST_ASSERT(dab < 1e-9);
}

void test_same_spin_and_onebody_controls() {
  std::vector<std::vector<double>> onebody, twobody;
  build_rdm(onebody, twobody);
  const size_t L = golden::NORB;
  const size_t n4 = L * L * L * L;
  TEST_ASSERT(max_abs_diff(twobody[0], golden::AA, n4) < 1e-9);
  TEST_ASSERT(max_abs_diff(onebody[0], golden::DM1A, L * L) < 1e-9);
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  std::vector<TestResult> results;

  results.push_back(run_test("2-RDM opposite-spin ab block vs PySCF",
                             test_opposite_spin_ab_block));
  results.push_back(run_test("2-RDM same-spin/1-RDM controls vs PySCF",
                             test_same_spin_and_onebody_controls));

  int failed = 0;
  for (const auto &result : results) {
    if (!result.passed) {
      ++failed;
      std::cout << "x FAIL: " << result.name << " (" << result.error_msg
                << ")\n";
    } else {
      std::cout << "o PASS: " << result.name << "\n";
    }
  }

  MPI_Finalize();
  return (failed == 0) ? 0 : 1;
}
