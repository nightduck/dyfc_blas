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

#ifndef DYFC_BLAS_ASUM_HPP
#define DYFC_BLAS_ASUM_HPP

#include "complex.hpp"
#include "prefixsum.hpp"
#include "types.hpp"

namespace dyfc {
namespace blas {

/**
 * Computes the sum of magnitudes of the vector elements, and returns a real scalar
 *
 * r = Σ |Re(x_i)| + |Im(x_i)|
 *
 * @tparam T The type of the elements in the vector. Supports any type with
 * defined arithmetic ops.
 * @tparam Par Number of elements retrieved in one read operation. Must be a
 * power of 2.
 *
 * @param[in]  n The length of the vectors x and y.
 * @param[in]  x The input vector to compute the sum of magnitudes of.
 * @param[out] result The output value to write to.
 */
template <typename T, const unsigned int Par = MAX_BITWIDTH / 8 / sizeof(T)>
void asum(unsigned int n, Vector<T, Par> &x, T &result) {
#pragma HLS INLINE
  ASSERT((n % Par) == 0, "n must be a multiple of Par");
  ASSERT(n == x.length(), "n must be equal to the length of x");
  ASSERT(x.read_lock(), "This vector is a pure stream and only accepts one reader");
  typename Vector<T, Par>::StreamType x_stream;
  x.read(x_stream);
  WideType<T, Par> x_val;
  WideType<T, Par> r_val;
  T r(0);
  for (unsigned int i = 0; i < n; i += Par) {
#pragma HLS PIPELINE
    x_val = x_stream.read();
    for (unsigned int j = 0; j < Par; j++) {
#pragma HLS UNROLL
      x_val[j] = abs(x_val[j]);  // TODO: Prefixsum works but asum doesn't. The abs unrolled loop is
                                 // the only difference. Investigate.
    }
    // TODO: Designate r_val as output and r as input.
    // TODO: Establish
    prefixsum<T, Par>(x_val, r_val, r);
    r = r_val[Par - 1];
  }
  result = r;
  ASSERT(x.empty(), "Vector x isn't empty");
}
// TODO: Specific implementations for the standard: sasum, dasum scasum, dzasum

}  // namespace blas
}  // namespace dyfc

#endif  // DYFC_BLAS_AXPY_HPP
