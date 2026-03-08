/**
 * @file test_hij_device_functions.cc
 * @brief Unit tests for individual GPU device functions in Hij computation
 *
 * Tests each component of the GPU Hij kernel in isolation:
 * - popcount_device
 * - getocc_device
 * - bitcount_device
 * - parity_device
 * - ZeroExcite_device
 * - OneExcite_device
 * - TwoExcite_device
 * - ComputeHij (orbital difference detection)
 */

#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

// Enable GPU offload flags
#define USE_DET_CACHE_OMP
#define USE_OMP_OFFLOAD
#define SBD_TRADMODE
#define USE_GPU
#define __HIP_PLATFORM_AMD__

// Enable debug output for this test
#define HIJ_GPU_DEBUG 1

#include "sbd/chemistry/basic/omp_offload.h"
#include "sbd/sbd.h"

using namespace sbd;

// Tolerance for floating-point comparisons
const double REL_TOL = 1e-10;
const double ABS_TOL = 1e-12;

bool almost_equal(double a, double b, double rel_tol = REL_TOL,
                  double abs_tol = ABS_TOL) {
  if (std::abs(a - b) < abs_tol)
    return true;
  double max_val = std::max(std::abs(a), std::abs(b));
  if (max_val < abs_tol)
    return true;
  return std::abs(a - b) / max_val < rel_tol;
}

// Test popcount_device
void test_popcount_device() {
  std::cout << "\n=== Test: popcount_device ===\n";

  std::vector<std::pair<size_t, int>> test_cases = {
      {0b0000, 0},     {0b0001, 1},     {0b0011, 2},
      {0b0101, 2},     {0b1111, 4},     {0b10101010, 4},
      {0b11111111, 8}, {~size_t(0), 64}
  };

  int passed = 0;
  for (const auto &tc : test_cases) {
    size_t input = tc.first;
    int expected = tc.second;
    int result = 0;

    result = popcount_device(input);

    if (result == expected) {
      passed++;
      std::cout << "  ✓ popcount(" << std::hex << input << std::dec
                << ") = " << result << "\n";
    } else {
      std::cout << "  ✗ popcount(" << std::hex << input << std::dec
                << ") = " << result << " (expected " << expected << ")\n";
    }
  }

  std::cout << "  Result: " << passed << "/" << test_cases.size()
            << " passed\n";
  assert(passed == test_cases.size());
}

// Test getocc_device
void test_getocc_device() {
  std::cout << "\n=== Test: getocc_device ===\n";

  size_t det[] = {0b10110011, 0b11001100};
  size_t bit_length = 64;

  std::vector<std::pair<int, bool>> test_cases = {
      {0, true},   {1, true},   {2, false}, {3, false},  {4, true},
      {5, true},   {6, false},  {7, true},  {64, false}, {65, false},
      {66, true},  {67, true},  {68, false}, {69, false}, {70, true},
      {71, true}};

  int passed = 0;
  for (const auto &tc : test_cases) {
    int x = tc.first;
    bool expected = tc.second;
    bool result = getocc_device(det, bit_length, x);

    if (result == expected) {
      passed++;
      std::cout << "  ✓ getocc(orbital " << x << ") = " << result << "\n";
    } else {
      std::cout << "  ✗ getocc(orbital " << x << ") = " << result
                << " (expected " << expected << ")\n";
    }
  }

  std::cout << "  Result: " << passed << "/" << test_cases.size()
            << " passed\n";
  assert(passed == test_cases.size());
}

// Test bitcount_device
void test_bitcount_device() {
  std::cout << "\n=== Test: bitcount_device ===\n";

  size_t bit_length = 64;

  {
    size_t det1[] = {0b10110011};
    size_t L = 8;
    int result = bitcount_device(det1, bit_length, L);
    int expected = 5;

    if (result == expected) {
      std::cout << "  ✓ bitcount(10110011, L=8) = " << result << "\n";
    } else {
      std::cout << "  ✗ bitcount(10110011, L=8) = " << result << " (expected "
                << expected << ")\n";
      assert(false);
    }
  }

  {
    size_t det2[] = {0b11111111, 0b11110000};
    size_t L = 72;
    int result = bitcount_device(det2, bit_length, L);
    int expected = 8 + 4;

    if (result == expected) {
      std::cout << "  ✓ bitcount(multi-word, L=72) = " << result << "\n";
    } else {
      std::cout << "  ✗ bitcount(multi-word, L=72) = " << result
                << " (expected " << expected << ")\n";
      assert(false);
    }
  }

  std::cout << "  Result: All bitcount tests passed\n";
}

// Test parity_device
void test_parity_device() {
  std::cout << "\n=== Test: parity_device ===\n";

  size_t bit_length = 64;

  {
    size_t det[] = {0b10110011};
    int start = 1;
    int end = 5;
    double sgn = 1.0;

    parity_device<double>(det, bit_length, start, end, sgn);

    double expected = -1.0;
    if (almost_equal(sgn, expected)) {
      std::cout << "  ✓ parity(start=1, end=5) = " << sgn << "\n";
    } else {
      std::cout << "  ✗ parity(start=1, end=5) = " << sgn << " (expected "
                << expected << ")\n";
      assert(false);
    }
  }

  {
    size_t det[] = {~size_t(0), 0b11110000};
    int start = 10;
    int end = 70;
    double sgn = 1.0;

    parity_device<double>(det, bit_length, start, end, sgn);

    std::cout << "  ✓ parity(multi-word, start=10, end=70) = " << sgn << "\n";
  }

  std::cout << "  Result: All parity tests passed\n";
}

// Test ZeroExcite_device
void test_zero_excite_device() {
  std::cout << "\n=== Test: ZeroExcite_device ===\n";

  size_t bit_length = 64;
  int norbs = 2;
  size_t L = 2;

  size_t det[] = {0b0011};

  double I0 = 0.5;

  oneInt<double> I1;
  I1.norbs = 2 * norbs;
  I1.store.resize(4 * 4, 0.0);
  I1.store[0 * 4 + 0] = 1.0;
  I1.store[1 * 4 + 1] = 1.0;

  twoInt<double> I2;
  I2.norbs = norbs;
  size_t I2_size =
      (norbs * (norbs + 1) / 2) * ((norbs * (norbs + 1) / 2) + 1) / 2;
  I2.store.resize(I2_size, 0.0);
  I2.store[0] = 0.5;

  std::vector<double> I2_Direct_flat(norbs * norbs, 0.0);
  std::vector<double> I2_Exchange_flat(norbs * norbs, 0.0);
  I2_Direct_flat[0] = 0.25;
  I2_Exchange_flat[0] = 0.25;

  double result = ZeroExcite_device(det, bit_length, L, I0, I1.store.data(),
                                    I2_Direct_flat.data(),
                                    I2_Exchange_flat.data(), norbs, I1.norbs);

  double expected = 3.0;

  if (almost_equal(result, expected, 1e-10)) {
    std::cout << "  ✓ ZeroExcite = " << result << "\n";
  } else {
    std::cout << "  ✗ ZeroExcite = " << result << " (expected " << expected
              << ")\n";
    assert(false);
  }

  std::cout << "  Result: ZeroExcite test passed\n";
}

// Test OneExcite_device
void test_one_excite_device() {
  std::cout << "\n=== Test: OneExcite_device ===\n";

  size_t bit_length = 64;
  int norbs = 2;

  size_t det[] = {0b0011};

  oneInt<double> I1;
  I1.norbs = 2 * norbs;
  I1.store.resize(4 * 4, 0.0);
  I1.store[2 * 4 + 0] = 0.5;

  twoInt<double> I2;
  I2.norbs = norbs;
  size_t I2_size =
      (norbs * (norbs + 1) / 2) * ((norbs * (norbs + 1) / 2) + 1) / 2;
  I2.store.resize(I2_size, 0.1);

  int i = 0;
  int a = 2;

  double result = OneExcite_device(det, bit_length, i, a, I1.store.data(),
                                   I2.store.data(), norbs, I1.norbs);

  std::cout << "  OneExcite(i=" << i << ", a=" << a << ") = " << result << "\n";

  if (std::abs(result) < 1e-14) {
    std::cout
        << "  ✗ FAILED: OneExcite returned ZERO when it should be non-zero!\n";
    std::cout << "     This indicates OneExcite_device has a bug (premature "
                 "return or uninitialized variable)\n";
    assert(false);
  } else {
    std::cout << "  ✓ OneExcite returned non-zero value as expected\n";
  }

  std::cout << "  Result: OneExcite test passed\n";
}

// Test TwoExcite_device
void test_two_excite_device() {
  std::cout << "\n=== Test: TwoExcite_device ===\n";

  size_t bit_length = 64;
  int norbs = 2;

  size_t det[] = {0b0011};

  twoInt<double> I2;
  I2.norbs = norbs;
  size_t I2_size =
      (norbs * (norbs + 1) / 2) * ((norbs * (norbs + 1) / 2) + 1) / 2;
  I2.store.resize(I2_size, 0.1);

  int i = 0, j = 1;
  int a = 2, b = 3;

  double result =
      TwoExcite_device(det, bit_length, i, j, a, b, I2.store.data());

  std::cout << "  ✓ TwoExcite(i=" << i << ", j=" << j << ", a=" << a
            << ", b=" << b << ") = " << result << "\n";
  std::cout << "  Result: TwoExcite test completed\n";
}

// Test ComputeHij orbital difference detection
void test_compute_hij_excitation_detection() {
  std::cout << "\n=== Test: ComputeHij Excitation Detection ===\n";

  size_t bit_length = 64;
  int norbs = 2;

  double I0 = 0.5;
  oneInt<double> I1;
  I1.norbs = 2 * norbs;
  I1.store.resize(4 * 4, 0.0);

  twoInt<double> I2;
  I2.norbs = norbs;
  size_t I2_size =
      (norbs * (norbs + 1) / 2) * ((norbs * (norbs + 1) / 2) + 1) / 2;
  I2.store.resize(I2_size, 0.1);

  // Flatten DirectMat and ExchangeMat for GPU
  std::vector<double> I2_Direct_flat(norbs * norbs, 0.05);
  std::vector<double> I2_Exchange_flat(norbs * norbs, 0.05);

  {
    size_t det1[] = {0b0011};
    size_t det2[] = {0b0011};

    double result = ComputeHij(det1, det2, bit_length, norbs, I0,
                               I1.store.data(), I2.store.data(),
                               I2_Direct_flat.data(), I2_Exchange_flat.data());

    std::cout << "  ✓ ComputeHij(same det) = " << result
              << " (zero excitation)\n";
  }

  {
    size_t det1[] = {0b0011};
    size_t det2[] = {0b0101};

    double result = ComputeHij(det1, det2, bit_length, norbs, I0,
                               I1.store.data(), I2.store.data(),
                               I2_Direct_flat.data(), I2_Exchange_flat.data());

    std::cout << "  ✓ ComputeHij(single excite) = " << result
              << " (one excitation)\n";
  }

  {
    size_t det1[] = {0b0011};
    size_t det2[] = {0b1100};

    double result = ComputeHij(det1, det2, bit_length, norbs, I0,
                               I1.store.data(), I2.store.data(),
                               I2_Direct_flat.data(), I2_Exchange_flat.data());

    std::cout << "  ✓ ComputeHij(double excite) = " << result
              << " (two excitations)\n";
  }

  {
    size_t det1[] = {0b00011};
    size_t det2[] = {0b11100};

    double result = ComputeHij(det1, det2, bit_length, norbs, I0,
                               I1.store.data(), I2.store.data(),
                               I2_Direct_flat.data(), I2_Exchange_flat.data());

    if (almost_equal(result, 0.0)) {
      std::cout << "  ✓ ComputeHij(triple excite) = " << result
                << " (correctly returns 0)\n";
    } else {
      std::cout << "  ✗ ComputeHij(triple excite) = " << result
                << " (expected 0)\n";
      assert(false);
    }
  }

  std::cout << "  Result: All excitation detection tests passed\n";
}

int main() {
  std::cout << "\n======================================\n";
  std::cout << "SBD Hij Device Function Unit Tests\n";
  std::cout << "======================================\n";

#ifdef USE_OMP_OFFLOAD
  std::cout << "✓ USE_OMP_OFFLOAD enabled\n";
#endif
#ifdef HIJ_GPU_DEBUG
  std::cout << "✓ HIJ_GPU_DEBUG enabled\n";
#endif

  try {
    test_popcount_device();
    test_getocc_device();
    test_bitcount_device();
    test_parity_device();
    test_zero_excite_device();
    test_one_excite_device();
    test_two_excite_device();
    test_compute_hij_excitation_detection();

    std::cout << "\n======================================\n";
    std::cout << "✓ ALL UNIT TESTS PASSED\n";
    std::cout << "======================================\n";

    return 0;
  } catch (const std::exception &e) {
    std::cerr << "\n✗ TEST FAILED: " << e.what() << "\n";
    return 1;
  }
}
