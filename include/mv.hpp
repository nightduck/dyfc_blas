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

#ifndef DYFC_BLAS_MV_HPP
#define DYFC_BLAS_MV_HPP

#include "complex.hpp"
#include "prefixsum.hpp"
#include "types.hpp"

namespace dyfc {
namespace blas {

/**
 * Computes a matrix-vector product.
 *
 * r = alpha * A * x
 *
 * @tparam T The type of the elements in the matrix and vectors. Supports any
 * type with defined arithmetic ops.
 * @tparam Order The order of the matrix. Must be either RowMajor or ColMajor.
 * @tparam Par Number of elements retrieved in one read operation. Must be a
 * power of 2.
 *
 * @param[in]  m The number of rows in the matrix A, and the size of the vectors y and r.
 * @param[in]  n The number of cols in the matrix A, and the size of the input vector x. Ignored if
 *             A is Triangular, Symmetric, or Hermitian, as these are square matrices.
 * @param[in]  alpha The scalar to multiply the input matrix A by.
 * @param[in]  A The input matrix to multiply.
 * @param[in]  x The input vector to multiply.
 * @param[out] result The output vector to write to.
 * @param[in]  buffer A buffer of size m to store the intermediate results of the
 */
template <typename T, const MajorOrder Order = RowMajor,
          const unsigned int Par = MAX_BITWIDTH / 8 / sizeof(T), const unsigned int Par2 = Par>
void mv(const unsigned int m, const unsigned int n, T alpha, Matrix<T, Order, Par> &A,
        Vector<T, Par> &x, Vector<T, Par2> &result, T *buffer = nullptr) {
#pragma HLS INLINE
#ifndef __SYNTHESIS__
  assert((n % Par) == 0);
  assert((m % Par) == 0);
  assert((m % Par2) == 0);
  assert(n == A.cols());
  assert(m == A.rows());
  assert(n == x.length());
  assert(m == result.length());
  assert(("This matrix is a pure stream and only accepts one reader", A.read_lock()));
  assert(("This vector is a pure stream and only accepts one reader", x.read_lock()));
  assert(("This vector only accepts one writer", result.write_lock()));
  assert(("If A is ColMajor, a buffer of size M must be provided",
          Order == RowMajor || buffer != nullptr));
#endif
  typename Matrix<T, Order, Par>::StreamType A_stream;
  typename Vector<T, Par>::StreamType x_stream;
  A.read(A_stream);
  if (Order == RowMajor) {
    x.read(x_stream, 1, A.rows());
    T r = 0;
    WideType<T, Par2> r_out;
  LOOP_gemv_rm:
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j += Par) {
#pragma HLS LOOP_FLATTEN
#pragma HLS PIPELINE
        WideType<T, Par> m_val = A_stream.read();
        WideType<T, Par> v_val = x_stream.read();
        WideType<T, Par> r_val = T(0);
        WideType<T, Par> rsum_val = T(0);
      LOOP_multiply_elements:
        for (int k = 0; k < Par; k++) {
#pragma HLS UNROLL
          r_val[k] = alpha * m_val[k] * v_val[k];
        }
        prefixsum<T, Par>(r_val, rsum_val, r);
        r = rsum_val[Par - 1];
        if (j + Par >= n) {
          r_out[i % Par2] = r;
          if (i % Par2 == Par2 - 1) {
            result.write(r_out);
          }
          r = T(0);
        }
      }
    }
  } else if (Order == ColMajor) {
    x.read(x_stream);
    // typename Vector<T, Par>::StreamType ring_buffer_stream;
    //     WideType<T, Par> ring_buffer[m / Par];
    // #pragma HLS ARRAY_PARTITION variable=ring_buffer complete dim=1
    WideType<T, Par> v_val = T(0);
    WideType<T, Par2> rout_val;

  LOOP_gemv_cm:
    for (size_t i = 0; i < n; i += Par) {
      for (size_t j = 0; j < Par; j++) {
        for (size_t k = 0; k < m; k += Par) {
#pragma HLS LOOP_FLATTEN
#pragma HLS PIPELINE
          WideType<T, Par> m_val = A_stream.read();
          WideType<T, Par> r_val;
          WideType<T, Par> rr_val = T(0);  // Running sum of these rows
          if (i + j > 0) {
            // rr_val = ring_buffer_stream.read();
            for (int l = 0; l < Par; l++) {
#pragma HLS UNROLL
              rr_val[l] = buffer[k + l];
            }
          }
          if (k == 0 && j == 0) {
            v_val = x_stream.read();
          }
        LOOP_gemv_cm_inner:
          for (int s = 0; s < Par; s++) {
#pragma HLS UNROLL
            r_val[s] = alpha * m_val[s] * v_val[j] + rr_val[s];
          }
          if (i + j < n - 1) {
            for (int l = 0; l < Par; l++) {
#pragma HLS UNROLL
              buffer[k + l] = r_val[l];
            }
          } else {
            for (int l = 0; l < Par; l++) {
#pragma HLS UNROLL factor = Par2
              rout_val[(k + l) % Par2] = r_val[l];
              if ((k + l) % Par2 == Par2 - 1) {
                result.write(rout_val);
              }
            }
          }
        }
      }
    }
  } else {
// There shouldn't be any other option
#ifndef __SYNTHESIS__
    assert(("Invalid MajorOrder option (this shouldn't be possible, wtf did you do?)", false));
#endif
  }

#ifndef __SYNTHESIS__
  assert(("Matrix A isn't empty", A.empty()));
  assert(("Vector x isn't empty", x.empty()));
  assert(("Vector result is empty", !result.empty()));
#endif
}
// TODO: Subtemplate for trmv, tbmv, tpmv
// TODO: Specific implementations for the standard: strmv, dtrmv, ctrmv, ztrmv,
// stbmv, dtbmv, ctbmv, ztbmv,
//       stpmv, dtpmv, ctpmv, ztpmv, ctpmv, ztpmv

/**
 * Computes a matrix-vector product and adds the result to a vector.
 *
 * r = alpha * A * x + beta * y
 *
 * @tparam T The type of the elements in the matrix and vectors. Supports any
 * type with defined arithmetic ops.
 * @tparam Order The order of the matrix A. Either RowMajor or ColMajor.
 * @tparam Par Number of elements retrieved in one read operation. Must be a
 * power of 2.
 *
 * @param[in]  m The number of rows in the matrix A, and the size of the output vector r.
 * @param[in]  n The number of cols in the matrix A, and the size of the input vector x.
 * @param[in]  alpha The scalar to multiply the input matrix A by.
 * @param[in]  A The input matrix to multiply.
 * @param[in]  x The input vector to multiply.
 * @param[in]  beta The scalar to multiply the input vector y by.
 * @param[in]  y The input vector to add to the result.
 * @param[out] result The output vector to write to.
 */
template <typename T, const MajorOrder Order = RowMajor,
          const unsigned int Par = MAX_BITWIDTH / 8 / sizeof(T), const unsigned int Par2 = Par>
void mv(const unsigned int m, const unsigned int n, T alpha, Matrix<T, Order, Par> &A,
        Vector<T, Par> &x, T beta, Vector<T, Par> &y, Vector<T, Par2> &result,
        T *buffer = nullptr) {
#pragma HLS INLINE
#ifndef __SYNTHESIS__
  assert((n % Par) == 0);
  assert((m % Par) == 0);
  assert(n == A.cols());
  assert(m == A.rows());
  assert(n == x.length());
  assert(m == y.length());
  assert(m == result.length());
  assert(("If A is ColMajor, a buffer of size M must be provided",
          Order == RowMajor || buffer != nullptr));
#endif
  Vector<T, Par> Ax(m);
  mv<T, Order, Par, Par2>(m, n, alpha, A, x, Ax, buffer);
  axpy<T, Par>(m, beta, y, Ax, result);

#ifndef __SYNTHESIS__
  assert(("Matrix A isn't empty", A.empty()));
  assert(("Vector x isn't empty", x.empty()));
  assert(("Vector y isn't empty", y.empty()));
  assert(("Intermediary vector Ax isn't empty", Ax.empty()));
  assert(("Vector result is empty", !result.empty()));
#endif
}
// TODO: Subtemplates for gemv, hemv, symv, gbmv, hbmv, sbmv, hpmv, spmv
// TODO: Specific implementations for the standard: cgemv, dgemv, sgemv, zgemv,
// chemv, zhemv,
//       ssymv, dsymv, sgbmv, dgbmv, cgbmv, zgbmv, chbmv, zhbmv, ssbmv, dsbmv,
//       chpmv, zhpmv, sspmv, dspmv
// TODO: Original standard also takes options for if the matrix should be
// transposed (if general
//       or triangular), or if the diagonal is a unit
// TODO: The triangular cases are not specified in the standard with an addition
// operation, but
//       they are useful to implement anyways. Don't rename them to trmv, etc.

}  // namespace blas
}  // namespace dyfc

#endif  // DYFC_BLAS_MV_HPP
