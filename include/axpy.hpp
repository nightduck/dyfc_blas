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

#ifndef DYFC_BLAS_AXPY_HPP
#define DYFC_BLAS_AXPY_HPP

#include "complex.hpp"
#include "types.hpp"

namespace dyfc {
namespace blas {

/**
 * Computes a vector-scalar product and adds the result to a vector.
 *
 * r = alpha * x + y
 *
 * @tparam T The type of the elements in the vector. Supports any type with defined arithmetic ops.
 * @tparam Par Number of elements retrieved in one read operation. Must be a power of 2.
 *
 * @param[in]  n The length of the vectors x and y.
 * @param[in]  alpha The scalar to multiply the input vector x by.
 * @param[in]  x The input vector to multiply.
 * @param[in]  y The input vector to add to the result.
 * @param[out] result The output vector to write to.
 */
template <typename T, unsigned int Par>
void axpy(unsigned int n, T alpha, Vector<T, Par> &x, Vector<T, Par> &y, Vector<T, Par> &result) {
#pragma HLS INLINE

#ifndef __SYNTHESIS__
  assert((n % Par) == 0);
#endif
  for (unsigned int i = 0; i < n; i += Par) {
#pragma HLS PIPELINE
    WideType<T, Par> r_val = T(0);
    WideType<T, Par> x_val = x.read();
    WideType<T, Par> y_val = y.read();
    for (int i = 0; i < Par; i++) {
#pragma HLS UNROLL
      r_val[i] = alpha * x_val[i] + y_val[i];
    }
    result.write(r_val);
  }
}
// TODO: Specific implementations for the standard: caxpy, daxpy, saxpy, zaxpy

}  // namespace blas
}  // namespace dyfc

#endif  // DYFC_BLAS_AXPY_HPP
