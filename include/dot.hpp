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

#ifndef DYFC_BLAS_DOT_HPP
#define DYFC_BLAS_DOT_HPP

#include "complex.hpp"
#include "prefixsum.hpp"
#include "types.hpp"

namespace dyfc {
namespace blas {

/**
 * Computes a vector-vector dot product.
 *
 * r = x . y
 *
 * @tparam T The type of the elements in the vector. Supports any type with
 * defined arithmetic ops.
 * @tparam Par Number of elements retrieved in one read operation. Must be a
 * power of 2.
 *
 * @param[in]  n The length of the vectors x and y.
 * @param[in]  x The input vector to multiply.
 * @param[in]  y The input vector to multiply.
 * @param[out] result The output value to write to.
 */
template <typename T, const unsigned int Par = MAX_BITWIDTH / 8 / sizeof(T)>
void dot(unsigned int n, Vector<T, Par> &x, Vector<T, Par> &y, T &result) {
#pragma HLS INLINE
#ifndef __SYNTHESIS__
  assert((n % Par) == 0);
  assert(n == x.length());
  assert(n == y.length());
  assert(("This vector is a pure stream and only accepts one reader", x.read_lock()));
  assert(("This vector is a pure stream and only accepts one reader", y.read_lock()));
#endif
  typename Vector<T, Par>::StreamType x_stream;
  typename Vector<T, Par>::StreamType y_stream;
  x.read(x_stream);
  y.read(y_stream);
  WideType<T, Par> x_val;
  WideType<T, Par> y_val;
  WideType<T, Par> xy_val;
  WideType<T, Par> r_val;
  T rr = 0;
  for (unsigned int i = 0; i < n; i += Par) {
#pragma HLS PIPELINE
    x_val = x_stream.read();
    y_val = y_stream.read();
    xy_val = 0;
    for (unsigned int j = 0; j < Par; j++) {
#pragma HLS UNROLL
      xy_val[j] = x_val[j] * y_val[j];
    }
    prefixsum<T, Par>(xy_val, r_val, rr);
    rr = r_val[Par - 1];
  }
  result = rr;
#ifndef __SYNTHESIS__
  assert(("Vector x isn't empty", x.empty()));
  assert(("Vector y isn't empty", y.empty()));
#endif
}

}  // namespace blas
}  // namespace dyfc

#endif  // DYFC_BLAS_DOT_HPP
