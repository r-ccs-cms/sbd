/**
@file sbd/chemistry/basic/omp_offload.h
@brief OpenMP target offload implementation of batched Hij computation
*/
#ifndef SBD_CHEMISTRY_BASIC_OMP_OFFLOAD_H
#define SBD_CHEMISTRY_BASIC_OMP_OFFLOAD_H

#include "sbd/framework/type_def.h"

#ifdef USE_OMP_OFFLOAD

#include <limits> // For std::numeric_limits (bisection test)

  // one-electron and two-electron integrals
  size_t I1_size, I2_size, I2_Direct_size, I2_Exchange_size;
  double *I1_ptr;
  double *I2_ptr;
  double *I2_Direct_ptr;
  double *I2_Exchange_ptr;

namespace sbd {

// Declare all device functions for OpenMP target offload
#pragma omp declare target

// Device-side parity computation (5-parameter version)
template <typename ElemT>
inline void parity_device(const size_t *dets, size_t bit_length, int start,
                          int end, double &sgn) {
  if (start > end)
    return;

  size_t blockStart = start / bit_length;
  size_t bitStart = start % bit_length;
  size_t blockEnd = end / bit_length;
  size_t bitEnd = end % bit_length;

  int nonZeroBits = 0;

  if (blockStart == blockEnd) {
    size_t mask = ((size_t(1) << bitEnd) - 1) ^ ((size_t(1) << bitStart) - 1);
    nonZeroBits += __builtin_popcountl(dets[blockStart] & mask);
  } else {
    if (bitStart != 0) {
      size_t mask = ~((size_t(1) << bitStart) - 1);
      nonZeroBits += __builtin_popcountl(dets[blockStart] & mask);
      blockStart++;
    }

    for (size_t i = blockStart; i < blockEnd; i++) {
      nonZeroBits += __builtin_popcountl(dets[i]);
    }

    if (bitEnd != 0) {
      size_t mask = (size_t(1) << bitEnd) - 1;
      nonZeroBits += __builtin_popcountl(dets[blockEnd] & mask);
    }
  }

  sgn *= (-2.0 * (nonZeroBits % 2) + 1.0);

  if ((dets[start / bit_length] >> (start % bit_length)) & 1) {
    sgn *= -1.0;
  }
}

// Device-side helper to check bit occupation
inline bool getocc_device(const size_t *det, size_t bit_length, int x) {
  size_t index = x / bit_length;
  size_t bit_pos = x % bit_length;
  return (det[index] >> bit_pos) & 1;
}

// Device-side helper to count set bits in determinant
inline int bitcount_device(const size_t *det, size_t bit_length, size_t L) {
  int count = 0;
  size_t full_words = L / bit_length;
  size_t remaining_bits = L % bit_length;

  for (size_t i = 0; i < full_words; ++i) {
    count += __builtin_popcountl(det[i]);
  }

  if (remaining_bits > 0) {
    size_t mask = (size_t(1) << remaining_bits) - 1;
    count += __builtin_popcountl(det[full_words] & mask);
  }

  return count;
}

// Device-side two-electron integral lookup with spin checking
// Mimics twoInt::Value(i,j,k,l) from integrals.h
// Returns 0.0 if spin checks fail: ((i%2 == j%2) && (k%2 == l%2))
// Otherwise computes compact index and returns I2[index]
template <typename ElemT>
inline ElemT twoInt_device(int i, int j, int k, int l, const ElemT *I2) {
  // Spin check: i and j must have same spin, k and l must have same spin
  if ((i & 1) != (j & 1) || (k & 1) != (l & 1)) {
    return ElemT(0);
  }

  // Extract spatial orbital indices (divide by 2)
  int I = i >> 1;
  int J = j >> 1;
  int K = k >> 1;
  int L = l >> 1;

  // Compute compact indices using triangular storage
  // ij_index = max(I,J)*(max(I,J)+1)/2 + min(I,J)
  int max_IJ = (I > J) ? I : J;
  int min_IJ = (I < J) ? I : J;
  int ij = max_IJ * (max_IJ + 1) / 2 + min_IJ;

  int max_KL = (K > L) ? K : L;
  int min_KL = (K < L) ? K : L;
  int kl = max_KL * (max_KL + 1) / 2 + min_KL;

  // Final compact storage index
  int a = (ij > kl) ? ij : kl;
  int b = (ij < kl) ? ij : kl;
  int idx = a * (a + 1) / 2 + b;

  return I2[idx];
}

// Device-side ZeroExcite evaluation
template <typename ElemT>
inline ElemT ZeroExcite_device(const size_t *det, size_t bit_length, size_t L,
                               const ElemT &I0, const ElemT *I1,
                               const ElemT *I2_Direct, const ElemT *I2_Exchange,
                               int norbs_spatial, int norbs_spin) {
  ElemT energy = I0;

  // Get closed orbitals
  int closed[256]; // Maximum 256 orbitals
  int num_closed = 0;

  for (int i = 0; i < 2 * L; i++) {
    if (getocc_device(det, bit_length, i)) {
      closed[num_closed++] = i;
    }
  }

  // One-electron contribution (use spin orbital indexing)
  for (int i = 0; i < num_closed; i++) {
    int I = closed[i];
    int idx = I * norbs_spin + I;
    ElemT h_val = I1[idx];
    energy += h_val;
  }

  // Two-electron contribution
  for (int i = 0; i < num_closed; i++) {
    int I = closed[i];
    for (int j = i + 1; j < num_closed; j++) {
      int J = closed[j];
      int orbital_I = I / 2;
      int orbital_J = J / 2;

      // Direct term (spin-independent) - use DirectMat accessor
      int idx = orbital_I + norbs_spatial * orbital_J;
      energy += I2_Direct[idx];

      // Exchange term (same spin only) - use ExchangeMat accessor
      if ((I % 2) == (J % 2)) {
        energy -= I2_Exchange[idx];
      }
    }
  }

  return energy;
}

// Device-side OneExcite evaluation
template <typename ElemT>
inline ElemT OneExcite_device(const size_t *det, size_t bit_length, int i,
                              int a, const ElemT *I1, const ElemT *I2, int norbs_spin) {
  double sgn = 1.0;
  parity_device<ElemT>(det, bit_length, (i < a) ? i : a, (i > a) ? i : a, sgn);
  int idx_ai = a * norbs_spin + i;
  ElemT energy = I1[idx_ai];
  int dsize = (norbs_spin + bit_length - 1) / bit_length;
  for (int x = 0; x < dsize; x++) {
    size_t bits = det[x];
    for (int pos = 0; pos < bit_length; pos++) {
       if ((bits & 1ULL) == 1ULL) {
          int j = x * bit_length + pos;
          // Use twoInt_device helper for cleaner code
          // I2.Value(a,i,j,j) - I2.Value(a,j,j,i)
          ElemT term1 = twoInt_device(a, i, j, j, I2);
          ElemT term2 = twoInt_device(a, j, j, i, I2);
          energy += term1 - term2;
        }
        bits >>= 1;
     }
  }
  return ElemT(sgn) * energy;
}

// Device-side TwoExcite evaluation
template <typename ElemT>
inline ElemT TwoExcite_device(const size_t *det, size_t bit_length, int i,
                              int j, int a, int b, const ElemT *I2) {
  double sgn = 1.0;
  int I = (i < j) ? i : j;
  int J = (i > j) ? i : j;
  int A = (a < b) ? a : b;
  int B = (a > b) ? a : b;

  parity_device<ElemT>(det, bit_length, (I < A) ? I : A, (I > A) ? I : A, sgn);
  parity_device<ElemT>(det, bit_length, (J < B) ? J : B, (J > B) ? J : B, sgn);

  if (A > J || B < I)
    sgn *= -1.0;

  // Use twoInt_device helper for proper spin checking
  // CPU code: I2.Value(A,I,B,J) - I2.Value(A,J,B,I)
  ElemT term1 = twoInt_device(A, I, B, J, I2);
  ElemT term2 = twoInt_device(A, J, B, I, I2);

  return ElemT(sgn) * (term1 - term2);
}

// Device-side determinant computation (TRADMODE version)
inline void ComputeDetFromAlphaBeta_TRADMODE(const size_t *A, const size_t *B,
                                             size_t *D, size_t bit_length,
                                             size_t L, size_t dsize) {
  int fsize = L / bit_length;
  int half = bit_length / 2;
  int extra = L - fsize * bit_length;
  int case1 = (extra > 0) && (extra <= half);
  int case2 = (extra > 0) && (extra > half);

  for (int j = 0; j < fsize; j++) {
    D[2 * j] = 0;
    D[2 * j + 1] = 0;
    for (int i = 0; i < half; i++) {
      if (A[j] & (1L << i))
        D[2 * j] |= (1L << (2 * i));
      if (B[j] & (1L << i))
        D[2 * j] |= (1L << (2 * i + 1));
      if (A[j] & (1L << (i + half)))
        D[2 * j + 1] |= (1L << (2 * i));
      if (B[j] & (1L << (i + half)))
        D[2 * j + 1] |= (1L << (2 * i + 1));
    }
  }

  if (case1) {
    int j = fsize;
    D[2 * j] = 0;
    for (int i = 0; i < extra; i++) {
      if (A[j] & (1L << i))
        D[2 * j] |= (1L << (2 * i));
      if (B[j] & (1L << i))
        D[2 * j] |= (1L << (2 * i + 1));
    }
  }

  if (case2) {
    int j = fsize;
    D[2 * j] = 0;
    D[2 * j + 1] = 0;
    for (int i = 0; i < half; i++) {
      if (A[j] & (1L << i))
        D[2 * j] |= (1L << (2 * i));
      if (B[j] & (1L << i))
        D[2 * j] |= (1L << (2 * i + 1));
    }
    for (int i = half; i < extra; i++) {
      if (A[j] & (1L << i))
        D[2 * j + 1] |= (1L << (2 * i - bit_length));
      if (B[j] & (1L << i))
        D[2 * j + 1] |= (1L << (2 * i + 1 - bit_length));
    }
  }
}

// Device-side determinant computation (Naive version)
inline void ComputeDetFromAlphaBeta_Naive(const size_t *A, const size_t *B,
                                          size_t *D, size_t bit_length,
                                          size_t L, size_t dsize) {
  // Initialize D to zero
  for (size_t i = 0; i < dsize; i++) {
    D[i] = 0;
  }

  for (size_t i = 0; i < L; i++) {
    size_t block = i / bit_length;
    size_t bit_pos = i % bit_length;
    size_t new_block_A = (2 * i) / bit_length;
    size_t new_bit_pos_A = (2 * i) % bit_length;
    size_t new_block_B = (2 * i + 1) / bit_length;
    size_t new_bit_pos_B = (2 * i + 1) % bit_length;

    if (A[block] & (size_t(1) << bit_pos)) {
      D[new_block_A] |= size_t(1) << new_bit_pos_A;
    }
    if (B[block] & (size_t(1) << bit_pos)) {
      D[new_block_B] |= size_t(1) << new_bit_pos_B;
    }
  }
}

#pragma omp end declare target

} // namespace sbd

#endif // USE_OMP_OFFLOAD

#endif // SBD_CHEMISTRY_BASIC_OMP_OFFLOAD_H
