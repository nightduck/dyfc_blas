// Copyright 2025 Oren Bell
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DYFC_BLAS_PREFIX_SUM_HPP
#define DYFC_BLAS_PREFIX_SUM_HPP

#include "complex.hpp"
#include "types.hpp"

namespace dyfc {
namespace blas {

/**
 * Computes a prefix sum from a vector of values, with an optional offset to add to all values.asum
 *
 * @tparam T The type of the elements in the vector. Officially only supports float, double,
 *           Complex<float>, and Complex<double>, but theoretically supports any type with
 *           defined arithmetic ops.
 * @tparam Par Number of elements retrieved in one read operation. Must be a power of 2.
 *
 * @param[in]  in The input vector to compute the prefix sum of.
 * @param[out] out The output vector to write to.
 * @param[in]  offset The value to add to all elements in the vector.
 */
template <typename T, const unsigned int Par>
void prefixsum(WideType<T, Par> &in, WideType<T, Par> &out, T &offset) {
#pragma HLS INLINE
  constexpr int LogPar = log2(Par);
  WideType<T, Par> x[LogPar + 1];
  // TODO: Combine write-in and write-out loops into main loop using conditionals
  for (int i = 0; i < Par; i++) {
#pragma HLS UNROLL
    x[0][i] = in[i];
  }
LOOP_WideType_hillis_steele:
  for (int i = 0; i <= LogPar; i++) {
    // #pragma HLS PIPELINE
    for (int j = 0; j < Par; j++) {
#pragma HLS UNROLL
      if (j >= 1 << i) {
        x[i + 1][j] = x[i][j] + x[i][j - (1 << i)];
      } else {
        x[i + 1][j] = x[i][j];
      }
    }
  }
  for (int i = 0; i < Par; i++) {
#pragma HLS UNROLL
    out[i] = x[LogPar][i] + offset;
  }
}

/**
 * Computes the prefix sum of a vector. Uses Hilles-Steele algorithm
 *
 * r_i = Σ x_i for i in [0, i]
 *
 * @tparam T The type of the elements in the vector. Supports any type with
 * defined arithmetic ops.
 * @tparam Par Number of elements retrieved in one read operation. Must be a
 * power of 2.
 *
 * @param[in]  n The length of the vectors x and y.
 * @param[in]  x The input vector to compute the prefix sum of.
 * @param[out] result The output vector to write to.
 */
template <typename T, const unsigned int Par>
void prefixsum(unsigned int n, Vector<T, Par> &x, Vector<T, Par> &result) {
#pragma HLS INLINE
#ifndef __SYNTHESIS__
  assert((n % Par) == 0);
#endif
  WideType<T, Par> x_val;
  WideType<T, Par> r_val;
  T r = 0;
  for (unsigned int i = 0; i < n; i += Par) {
#pragma HLS PIPELINE
    x_val = x.read();
    prefixsum<T, Par>(x_val, r_val, r);
    r = r_val[Par - 1];
    result.write(r_val);
  }
}

}  // namespace blas
}  // namespace dyfc

#endif  // DYFC_BLAS_PREFIX_SUM_HPP
