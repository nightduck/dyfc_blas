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
 * This once templated implementation accomodates the caxpy, daxpy, saxpy, and zaxpy calls
 *
 * @tparam T The type of the elements in the vector. Officially only supports float, double,
 *           Complex<float>, and Complex<double>, but theoretically supports any type with
 *           defined arithmetic ops.
 * @tparam Par Number of elements retrieved in one read operation. Must be a power of 2.
 *
 * @param[in]  n The length of the vectors x and y.
 * @param[in]  alpha The scalar to multiply the input vector x by.
 * @param[in]  x The input vector to multiply.
 * @param[in]  y The input vector to add to the result.
 * @param[out] result The output vector to write to.
 */
template <typename T, const unsigned int Par = MAX_BITWIDTH / 8 / sizeof(T)>
void axpy(unsigned int n, T alpha, Vector<T, Par> &x, Vector<T, Par> &y, Vector<T, Par> &result) {
#pragma HLS INLINE
#ifndef __SYNTHESIS__
  assert((n % Par) == 0);
  assert(n == x.length());
  assert(n == y.length());
  assert(n == result.length());
  assert(("This vector is a pure stream and only accepts one reader", x.read_lock()));
  assert(("This vector is a pure stream and only accepts one reader", y.read_lock()));
  assert(("This vector only accepts one writer", result.write_lock()));
#endif
  typename Vector<T, Par>::StreamType x_stream;
  typename Vector<T, Par>::StreamType y_stream;
  x.read(x_stream);
  y.read(y_stream);
  for (unsigned int i = 0; i < n; i += Par) {
#pragma HLS PIPELINE
    WideType<T, Par> r_val = T(0);
    WideType<T, Par> x_val = x_stream.read();
    WideType<T, Par> y_val = y_stream.read();
    for (int j = 0; j < Par; j++) {
#pragma HLS UNROLL
      r_val[j] = alpha * x_val[j] + y_val[j];
    }
    result.write(r_val);
  }
#ifndef __SYNTHESIS__
  assert(("Vector x isn't empty", x.empty()));
  assert(("Vector y isn't empty", y.empty()));
  assert(("Vector result is empty", !result.empty()));
#endif
}

template <typename T, const MajorOrder Order, const unsigned int Par = MAX_BITWIDTH / 8 / sizeof(T)>
void axpy(unsigned int n, unsigned int m, T alpha, Matrix<T, Order, Par> &x,
          Matrix<T, Order, Par> &y, Matrix<T, Order, Par> &result) {
#pragma HLS INLINE
#ifndef __SYNTHESIS__
  assert((n % Par) == 0);
  assert((m % Par) == 0);
  assert(n == x.rows());
  assert(m == x.cols());
  assert(n == y.rows());
  assert(m == y.cols());
  assert(n == result.rows());
  assert(m == result.cols());
  assert(("This matrix is a pure stream and only accepts one reader", x.read_lock()));
  assert(("This matrix is a pure stream and only accepts one reader", y.read_lock()));
  assert(("This matrix only accepts one writer", result.write_lock()));
#endif
  typename Matrix<T, Order, Par>::StreamType x_stream;
  typename Matrix<T, Order, Par>::StreamType y_stream;
  x.read(x_stream);
  y.read(y_stream);
  for (unsigned int i = 0; i < n; i++) {
    for (unsigned int j = 0; j < m; j += Par) {
#pragma HLS PIPELINE
      WideType<T, Par> r_val = T(0);
      WideType<T, Par> x_val = x_stream.read();
      WideType<T, Par> y_val = y_stream.read();
      for (int k = 0; k < Par; k++) {
#pragma HLS UNROLL
        r_val[k] = alpha * x_val[k] + y_val[k];
      }
      result.write(r_val);
    }
  }
#ifndef __SYNTHESIS__
  assert(("Matrix x isn't empty", x.empty()));
  assert(("Matrix y isn't empty", y.empty()));
  assert(("Matrix result is empty", !result.empty()));
#endif
}

}  // namespace blas
}  // namespace dyfc

#endif  // DYFC_BLAS_AXPY_HPP
