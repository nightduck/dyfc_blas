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

#ifndef DYFC_BLAS_EXAMPLE_HPP
#define DYFC_BLAS_EXAMPLE_HPP

#include "complex.hpp"
#include "types.hpp"

namespace dyfc {
namespace blas {

/**
 * This is not a part of the library, but a arbitrary function that is called by the EXAMPLE test.
 * This just ensures that the EXAMPLE test compiles and runs, to prevent confusion for anyone who
 * may accidentally try to build it
 * 
 * @tparam T The type of the elements in the vector. Supports any type with defined arithmetic ops.
 * @tparam Par Number of elements retrieved in one read operation. Must be a power of 2.
 * @tparam Order The major order of the matrix. Can be either RowMajor or ColMajor.
 * 
 * @param[in]  n The length of the vector x and the size of the square matrix A.
 * @param[in]  x The input vector to multiply.
 * @param[in]  A The input matrix to add to.
 * @param[out] r The output vector to write to.
 */
template <typename T, unsigned int Par, MajorOrder Order>
void example(unsigned int n, T alpha, Vector<T, Par> &x, Matrix<T, Par, Order> &A, Vector<T, Par> &r) {
#pragma HLS INLINE
  WideType<T, Par> r_val;
  for (unsigned int i = 0; i < n; i += Par) {
    WideType<T, Par> x_val = x.read();
    r_val = 0;
    for (unsigned int j = 0; j < Par; j++) {
      for (unsigned int k = 0; k < n; k += Par) {
        WideType<T, Par> A_val = A.read();
        for (unsigned int l = 0; l < Par; l++) {
          if (i + j == k + l) {
            r_val[j] = alpha * (x_val[j] + A_val[l]);
          }
        }
      }
    }
    r.write(r_val);
  }
}

}  // namespace blas
}  // namespace dyfc

#endif  // DYFC_BLAS_EXAMPLE_HPP
