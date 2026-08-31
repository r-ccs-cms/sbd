/**
 * @file test_determinants.cc
 * @brief Unit and regression tests for DetFromAlphaBeta.
 */

#include "sbd/sbd.h"
#include "utils.h"
#include <mpi.h>

using namespace sbd;

//=============================================================================
// DetFromAlphaBeta Unit Tests
//=============================================================================

void test_DetFromAlphaBeta_simple() {
  const size_t bit_length = 64;
  const size_t L = 4; // 4 orbitals

  std::vector<size_t> A = {0b0011}; // Alpha: |0011⟩
  std::vector<size_t> B = {0b0101}; // Beta:  |0101⟩

  auto D1 = DetFromAlphaBeta(A, B, bit_length, L);
  std::vector<size_t> D2((2 * L + bit_length - 1) / bit_length, 0);
  DetFromAlphaBeta(A, B, bit_length, L, D2);

  TEST_ASSERT(D1.size() == D2.size());
  for (size_t i = 0; i < D1.size(); i++) {
    TEST_ASSERT(D1[i] == D2[i]);
  }

  TEST_ASSERT((D1[0] & 0b11) == 0b11);
  TEST_ASSERT((D1[0] & 0b1100) == 0b0100);
  TEST_ASSERT((D1[0] & 0b110000) == 0b100000);
  TEST_ASSERT((D1[0] & 0b11000000) == 0);

  // Call with det_vector<size_t, det_kind::half>::row as inputs A and B.
  // det_vector(size_t n, const std::vector<ElemT>& v) constructs n rows each
  // initialised to v; [0] returns a row& (non-owning view into the flat buffer).
  det_vector<size_t, det_kind::half> dv_A(1, A);
  det_vector<size_t, det_kind::half> dv_B(1, B);
  auto D3 = DetFromAlphaBeta(dv_A[0], dv_B[0], bit_length, L);
  TEST_ASSERT(D3.size() == D1.size());
  for (size_t i = 0; i < D1.size(); i++) {
    TEST_ASSERT(D3[i] == D1[i]);
  }

  // Void form with det_vector<size_t, det_kind::full>::row as output D.
  const size_t dsize = D1.size();
  det_vector<size_t, det_kind::full> dv_D(1, std::vector<size_t>(dsize, 0));
  DetFromAlphaBeta(A, B, bit_length, L, dv_D[0]);
  for (size_t i = 0; i < dsize; i++) {
    TEST_ASSERT(dv_D[0][i] == D1[i]);
  }

  // Void form: det_vector inputs + det_vector full output.
  det_vector<size_t, det_kind::full> dv_D2(1, std::vector<size_t>(dsize, 0));
  DetFromAlphaBeta(dv_A[0], dv_B[0], bit_length, L, dv_D2[0]);
  for (size_t i = 0; i < dsize; i++) {
    TEST_ASSERT(dv_D2[0][i] == D1[i]);
  }
}

void test_DetFromAlphaBeta_multiword() {
  const size_t bit_length = 64;
  const size_t L = 100;

  std::vector<size_t> A((L + bit_length - 1) / bit_length, 0);
  for (size_t i = 0; i < L; i += 5) {
    A[i / bit_length] |= (size_t(1) << (i % bit_length));
  }

  std::vector<size_t> B((L + bit_length - 1) / bit_length, 0);
  for (size_t i = 0; i < L; i += 7) {
    B[i / bit_length] |= (size_t(1) << (i % bit_length));
  }

  auto D1 = DetFromAlphaBeta(A, B, bit_length, L);

  for (size_t orb = 0; orb < L; orb++) {
    size_t alpha_bit = 2 * orb;
    size_t beta_bit = 2 * orb + 1;
    bool alpha_set =
        (D1[alpha_bit / bit_length] >> (alpha_bit % bit_length)) & 1;
    bool beta_set = (D1[beta_bit / bit_length] >> (beta_bit % bit_length)) & 1;
    TEST_ASSERT(alpha_set == (orb % 5 == 0));
    TEST_ASSERT(beta_set == (orb % 7 == 0));
  }

  // det_vector inputs: for L=100, half-dets are 2 words; full-dets are 4 words.
  // Use distinct det_kind values (101/102) so their independent _elem_size statics
  // don't conflict with the det_kind::half/full instances in test_DetFromAlphaBeta_simple.
  det_vector<size_t, static_cast<det_kind>(101)> dv_A(1, A);
  det_vector<size_t, static_cast<det_kind>(101)> dv_B(1, B);
  auto D2 = DetFromAlphaBeta(dv_A[0], dv_B[0], bit_length, L);
  TEST_ASSERT(D2.size() == D1.size());
  for (size_t i = 0; i < D1.size(); i++) {
    TEST_ASSERT(D2[i] == D1[i]);
  }

  // Void form with det_vector<size_t, static_cast<det_kind>(102)>::row as output D.
  const size_t dsize = D1.size();
  det_vector<size_t, static_cast<det_kind>(102)> dv_D(1, std::vector<size_t>(dsize, 0));
  DetFromAlphaBeta(A, B, bit_length, L, dv_D[0]);
  for (size_t i = 0; i < dsize; i++) {
    TEST_ASSERT(dv_D[0][i] == D1[i]);
  }

  // Void form: det_vector(101) inputs + det_vector(102) output.
  det_vector<size_t, static_cast<det_kind>(102)> dv_D2(1, std::vector<size_t>(dsize, 0));
  DetFromAlphaBeta(dv_A[0], dv_B[0], bit_length, L, dv_D2[0]);
  for (size_t i = 0; i < dsize; i++) {
    TEST_ASSERT(dv_D2[0][i] == D1[i]);
  }
}

void test_DetFromAlphaBeta_edge_cases() {
  const size_t bit_length = 64;
  size_t L = 4;

  using V = std::vector<size_t>;
  // Empty
  TEST_ASSERT(DetFromAlphaBeta(V{0}, V{0}, bit_length, L)[0] == 0);
  // Full
  TEST_ASSERT(DetFromAlphaBeta(V{0b1111}, V{0b1111}, bit_length, L)[0] ==
              0b11111111);
  // Alpha only
  TEST_ASSERT(DetFromAlphaBeta(V{0b1010}, V{0}, bit_length, L)[0] == 0b01000100);
  // Beta only
  TEST_ASSERT(DetFromAlphaBeta(V{0}, V{0b1010}, bit_length, L)[0] == 0b10001000);
}

void test_DetFromAlphaBeta_regression_Fe4S4() {
  const size_t bit_length = 64;
  const size_t L = 36;
  std::vector<size_t> A = {0b111111111111111111111111}; // 24 electrons
  std::vector<size_t> B = {0b11111111111111111111};     // 20 electrons

  auto D1 = DetFromAlphaBeta(A, B, bit_length, L);

  int count = 0;
  for (size_t i = 0; i < D1.size(); i++) {
    count += __builtin_popcountll(D1[i]);
  }
  TEST_ASSERT(count == 24 + 20);
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  std::vector<TestResult> results;

  results.push_back(
      run_test("DetFromAlphaBeta - simple", test_DetFromAlphaBeta_simple));
  results.push_back(run_test("DetFromAlphaBeta - multiword",
                             test_DetFromAlphaBeta_multiword));
  results.push_back(run_test("DetFromAlphaBeta - edge cases",
                             test_DetFromAlphaBeta_edge_cases));
  results.push_back(run_test("DetFromAlphaBeta - Fe4S4 regression",
                             test_DetFromAlphaBeta_regression_Fe4S4));

  int failed = 0;
  for (const auto &result : results) {
    if (!result.passed) {
      failed++;
      std::cout << "✗ FAIL: " << result.name << " (" << result.error_msg
                << ")\n";
    } else {
      std::cout << "✓ PASS: " << result.name << "\n";
    }
  }

  MPI_Finalize();
  return (failed == 0) ? 0 : 1;
}
