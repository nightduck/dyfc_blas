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
 */
template <typename T, const MajorOrder Order = RowMajor, const unsigned int Par = MAX_BITWIDTH / 8 / sizeof(T)>
void mv(const unsigned int m, const unsigned int n, T alpha, Matrix<T, Order, Par> &A, Vector<T, Par> &x,
        Vector<T, Par> &result) {
#pragma HLS INLINE
#ifndef __SYNTHESIS__
  assert((n % Par) == 0);
  assert((m % Par) == 0);
  assert(A.cols() == n);
  assert(A.rows() == m);
  assert(x.shape() == n);
  assert(result.shape() == m);

#endif
  if (Order == RowMajor) {
  T r = 0;
  WideType<T, Par> r_out;
  hls::stream<WideType<T, Par>> ringbuf_x;
LOOP_gemv_rm:
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j += Par) {
#pragma HLS LOOP_FLATTEN
#pragma HLS PIPELINE
      WideType<T, Par> m_val = A.read();
      WideType<T, Par> v_val;
      WideType<T, Par> r_val = T(0);
      WideType<T, Par> rsum_val = T(0);
      if (i > 0) {
        v_val = ringbuf_x.read();
      } else {
        v_val = x.read();
      }
      if (i < m - 1) {
        ringbuf_x.write(v_val);
      }
    LOOP_multiply_elements:
      for (int k = 0; k < Par; k++) {
#pragma HLS UNROLL
        r_val[k] = alpha * m_val[k] * v_val[k];
      }
      prefixsum<T, Par>(r_val, rsum_val, r);
      r = rsum_val[Par - 1];
      if (j + Par >= n) {
        r_out[i % Par] = r;
        if (i % Par == Par - 1) {
          result.write(r_out);
        }
        r = T(0);
      }
    }
  }
  } else if (Order == ColMajor) {    
    WideType<T, Par> ring_buffer[m/Par];
    WideType<T, Par> v_val = T(0);

LOOP_gemv_cm:
    for (size_t i = 0; i < n; i++) {
      for (size_t j = 0; j < m; j+=Par) {
        WideType<T, Par> m_val = A.read();
        WideType<T, Par> r_val;
        WideType<T, Par> rr_val = T(0);  // Running sum of these rows
        if (i > 0) {
          // rr_val = ring_buffer.read();
          rr_val = ring_buffer[j/Par];
        }
        if (j == 0 && i % Par == 0) {
          v_val = x.read();
        }
        for (int k = 0; k < Par; k++) {
  #pragma HLS UNROLL
          r_val[k] = alpha * m_val[k] * v_val[i % Par] + rr_val[k];
        }
        if (i < n-1) {
          // ring_buffer.write(r_val);
          ring_buffer[j/Par] = r_val;
        } else {
          result.write(r_val);
        }
      }
    }
  } else {
    // There shouldn't be any other option
    #ifndef __SYNTHESIS__
      assert(("Invalid MajorOrder option (this shouldn't be possible, wtf did you do?)", false));
    #endif
  }
  return;
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
template <typename T, const MajorOrder Order = RowMajor, const unsigned int Par = MAX_BITWIDTH / 8 / sizeof(T)>
void mv(const unsigned int m, const unsigned int n, T alpha, Matrix<T, Order, Par> &A, Vector<T, Par> &x,
        T beta, Vector<T, Par> &y, Vector<T, Par> &result) {
#pragma HLS INLINE
#ifndef __SYNTHESIS__
  assert((n % Par) == 0);
  assert((m % Par) == 0);
  assert(A.cols() == n);
  assert(A.rows() == m);
  assert(x.shape() == n);
  assert(y.shape() == m);
  assert(result.shape() == m);
#endif
  Vector<T, Par> Ax(m);
  mv(m, n, alpha, A, x, Ax);
  axpy(m, beta, y, Ax, result);


  #ifndef __SYNTHESIS__
    assert(("Matrix isn't empty", A.size() == 0));
    assert(("Vector x isn't empty", x.size() == 0));
    assert(("Vector y isn't empty", y.size() == 0));
    assert(("Intermediary vector Ax isn't empty", Ax.size() == 0));
  #endif
  return;
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
