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
  ASSERT((n % Par) == 0, "n must be a multiple of Par");
  ASSERT(n == x.length(), "n must be equal to the length of x");
  ASSERT(n == y.length(), "n must be equal to the length of y");
  ASSERT(n == result.length(), "n must be equal to the length of result");
  ASSERT(x.read_lock(), "This vector is a pure stream and only accepts one reader");
  ASSERT(y.read_lock(), "This vector is a pure stream and only accepts one reader");
  ASSERT(result.write_lock(), "This vector only accepts one writer");

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

  ASSERT(x.empty(), "Vector x isn't empty");
  ASSERT(y.empty(), "Vector y isn't empty");
  ASSERT(!result.empty(), "Vector result is empty");
}

template <typename T, const MajorOrder Order, const unsigned int Par = MAX_BITWIDTH / 8 / sizeof(T)>
void axpy(unsigned int n, unsigned int m, T alpha, Matrix<T, Order, Par> &x,
          Matrix<T, Order, Par> &y, Matrix<T, Order, Par> &result) {
#pragma HLS INLINE
  ASSERT((n % Par) == 0, "n must be a multiple of Par");
  ASSERT((m % Par) == 0, "m must be a multiple of Par");
  ASSERT(n == x.rows(), "n must be equal to the number of rows of x");
  ASSERT(m == x.cols(), "m must be equal to the number of cols of x");
  ASSERT(n == y.rows(), "n must be equal to the number of rows of y");
  ASSERT(m == y.cols(), "m must be equal to the number of cols of y");
  ASSERT(x.read_lock(), "This matrix is a pure stream and only accepts one reader");
  ASSERT(y.read_lock(), "This matrix is a pure stream and only accepts one reader");
  ASSERT(result.write_lock(), "This matrix only accepts one writer");

  typename Matrix<T, Order, Par>::StreamType x_stream;
  typename Matrix<T, Order, Par>::StreamType y_stream;
  x.read(x_stream);
  y.read(y_stream);
  for (unsigned int i = 0; i < m; i++) {
    for (unsigned int j = 0; j < n; j += Par) {
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

  ASSERT(x.empty(), "Matrix x isn't empty");
  ASSERT(y.empty(), "Matrix y isn't empty");
  ASSERT(!result.empty(), "Matrix result is empty");
}

}  // namespace blas
}  // namespace dyfc

#endif  // DYFC_BLAS_AXPY_HPP
