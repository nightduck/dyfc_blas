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

#ifndef DYFC_BLAS_MM_HPP
#define DYFC_BLAS_MM_HPP

#include <hls_stream.h>

#include "complex.hpp"
#include "prefixsum.hpp"
#include "types.hpp"

namespace dyfc {
namespace blas {

/**
 * Performs matrix multiplication where A is row major and B is column major.
 * This is a helper function for mm and serves as the core implementation of the matrix
 * multiplication algorithm when A and B are different major orders.
 *
 * @tparam T The type of the elements in the matrix and vectors. Supports any
 * type with defined arithmetic ops.
 * @tparam Order The major order of the output matrix.
 * @tparam Par Number of elements retrieved in one read operation. Must be a
 * power of 2.
 *
 * @param[in]  m The number of rows in the matrix A and the output matrix result.
 * @param[in]  n The number of columns in the matrix B and the output matrix result.
 * @param[in]  k The number of columns in the matrix A and the number of rows in the matrix B.
 * @param[in]  alpha The scalar to multiply the input matrix A by.
 * @param[in]  A_stream Stream of elements from the first input matrix where each row has been
 * repeated k times.
 * @param[in]  B_stream Stream of elements from the second input matrix where the entire matrix has
 * been repeated m times.
 * @param[out] result The output matrix to write to.
 */
template <typename T, const MajorOrder Order, const unsigned int Par = MAX_BITWIDTH / 8 / sizeof(T)>
void mm_impl(const unsigned int m, const unsigned int n, const unsigned int k, T alpha,
               hls::stream<WideType<T, Par>> &A_stream, hls::stream<WideType<T, Par>> &B_stream,
               Matrix<T, Order, Par> &result) {
#pragma HLS INLINE
  T r(0);
  WideType<T, Par> r_out;
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      for (size_t l = 0; l < k; l += Par) {
#pragma HLS PIPELINE
        WideType<T, Par> a = A_stream.read();
        WideType<T, Par> b = B_stream.read();
        WideType<T, Par> r_val;
        WideType<T, Par> rsum_val = T(0);
        for (size_t p = 0; p < Par; p++) {
#pragma HLS UNROLL
          r_val[p] = a[p] * b[p];
        }
        prefixsum<T, Par>(r_val, rsum_val, r);
        r = rsum_val[Par - 1];
        if (l + Par >= k) {
          r_out[j % Par] = r;
          if (j % Par == Par - 1) {
            result.write(r_out * alpha);
          }
          r = T(0);
        }
      }
    }
  }
#ifndef __SYNTHESIS__
  assert(("A_stream isn't empty", A_stream.empty()));
  assert(("B_stream isn't empty", B_stream.empty()));
  assert(("Matrix result is empty", !result.empty()));
#endif
}

/**
 * Performs matrix multiplication where A is row major and B is column major.
 * This is a helper function for mm.
 *
 * @tparam T The type of the elements in the matrix and vectors. Supports any
 * type with defined arithmetic ops.
 * @tparam Par Number of elements retrieved in one read operation. Must be a
 * power of 2.
 *
 * @param[in]  m The number of rows in the matrix A and the output matrix result.
 * @param[in]  n The number of columns in the matrix B and the output matrix result.
 * @param[in]  k The number of columns in the matrix A and the number of rows in the matrix B.
 * @param[in]  alpha The scalar to multiply the input matrix A by.
 * @param[in]  A The input matrix to multiply.
 * @param[in]  B The input matrix to multiply.
 * @param[out] result The output matrix to write to.
 * @param[in]  buffer Unused.
 */
template <typename T, const unsigned int Par = MAX_BITWIDTH / 8 / sizeof(T)>
void mm(const unsigned int m, const unsigned int n, const unsigned int k, T alpha,
        Matrix<T, RowMajor, Par> &A, Matrix<T, ColMajor, Par> &B, Matrix<T, RowMajor, Par> &result,
        T *buffer = nullptr) {
#pragma HLS INLINE
#ifndef __SYNTHESIS__
  assert((n % Par) == 0);
  assert((m % Par) == 0);
  assert((k % Par) == 0);
  assert(A.rows() == m);
  assert(A.cols() == k);
  assert(B.rows() == k);
  assert(B.cols() == n);
  assert(result.rows() == m);
  assert(result.cols() == n);
  assert(("This matrix is a pure stream and only accepts one reader", A.read_lock()));
  assert(("This matrix is a pure stream and only accepts one reader", B.read_lock()));
  assert(("This matrix only accepts one writer", result.write_lock()));
#endif
  typename Matrix<T, RowMajor, Par>::StreamType A_stream;
  typename Matrix<T, ColMajor, Par>::StreamType B_stream;
  A.read(A_stream, false, B.cols(), 1);
  B.read(B_stream, false, 1, A.rows());

  mm_impl<T, RowMajor, Par>(m, n, k, alpha, A_stream, B_stream, result);
#ifndef __SYNTHESIS__
  assert(("Matrix A isn't empty", A.empty()));
  assert(("Matrix B isn't empty", B.empty()));
  assert(("Matrix result is empty", !result.empty()));
  assert(("A_stream isn't empty", A_stream.empty()));
  assert(("B_stream isn't empty", B_stream.empty()));
#endif
}

/**
 * Performs matrix multiplication where A is row major, B is column major, and C is column major.
 *
 * This leverages the identity that (AB)^T = B^T A^T to then
 *
 * @tparam T The type of the elements in the matrix and vectors. Supports any
 * type with defined arithmetic ops.
 * @tparam Par Number of elements retrieved in one read operation. Must be a
 * power of 2.
 *
 * @param[in]  m The number of rows in the matrix A and the output matrix result.
 * @param[in]  n The number of columns in the matrix B and the output matrix result.
 * @param[in]  k The number of columns in the matrix A and the number of rows in the matrix B.
 * @param[in]  alpha The scalar to multiply the input matrix A by.
 * @param[in]  A The input matrix to multiply.
 * @param[in]  B The input matrix to multiply.
 * @param[out] result The output matrix to write to.
 * @param[in]  buffer Unused.
 */
template <typename T, const unsigned int Par = MAX_BITWIDTH / 8 / sizeof(T)>
void mm(const unsigned int m, const unsigned int n, const unsigned int k, T alpha,
        Matrix<T, RowMajor, Par> &A, Matrix<T, ColMajor, Par> &B, Matrix<T, ColMajor, Par> &result,
        T *buffer = nullptr) {
#pragma HLS INLINE
#ifndef __SYNTHESIS__
  assert((n % Par) == 0);
  assert((m % Par) == 0);
  assert((k % Par) == 0);
  assert(A.rows() == m);
  assert(A.cols() == k);
  assert(B.rows() == k);
  assert(B.cols() == n);
  assert(result.rows() == m);
  assert(result.cols() == n);
  assert(("This matrix is a pure stream and only accepts one reader", A.read_lock()));
  assert(("This matrix is a pure stream and only accepts one reader", B.read_lock()));
  assert(("This matrix only accepts one writer", result.write_lock()));
#endif
  typename Matrix<T, RowMajor, Par>::StreamType A_stream;
  typename Matrix<T, ColMajor, Par>::StreamType B_stream;
  B.read(B_stream, false, A.rows(), 1);
  A.read(A_stream, false, 1, B.cols());
  mm_impl<T, ColMajor, Par>(n, m, k, alpha, B_stream, A_stream, result);
#ifndef __SYNTHESIS__
  assert(("Matrix A isn't empty", A.empty()));
  assert(("Matrix B isn't empty", B.empty()));
  assert(("Matrix result is empty", !result.empty()));
  assert(("Matrix result is unexpected size", result.size() == m * n / Par));
#endif
}

/**
 * Performs matrix multiplication
 *
 * r = alpha * A * B + beta * C
 *
 * @tparam T The type of the elements in the matrix and vectors. Supports any
 * type with defined arithmetic ops.
 * @tparam OrderA The major order of the matrix A. Can be either RowMajor or
 * ColMajor.
 * @tparam OrderB The major order of the matrix B. Can be either RowMajor or
 * ColMajor.
 * @tparam UpLo Whether the matrix is upper or lower triangular. This is ignored
 * for general matrices
 * @tparam Par Number of elements retrieved in one read operation. Must be a
 * power of 2.
 *
 * @param[in]  m The number of rows in the matrix A and the output matrix result.
 * @param[in]  n The number of columns in the matrix B and the output matrix result.
 * @param[in]  k The number of columns in the matrix A and the number of rows in the matrix B.
 * @param[in]  alpha The scalar to multiply the input matrix A by.
 * @param[in]  A The input matrix to multiply.
 * @param[in]  B The input matrix to multiply.
 * @param[in]  beta The scalar to multiply the input matrix C by.
 * @param[in]  C The input matrix to add to.
 * @param[out] result The output matrix to write to.
 */
template <typename T, const MajorOrder OrderA = RowMajor, const MajorOrder OrderB = ColMajor,
          const MajorOrder OrderC = RowMajor, const UpperLower UpLo = Upper,
          const unsigned int Par = MAX_BITWIDTH / 8 / sizeof(T)>
void mm(const unsigned int m, const unsigned int n, const unsigned int k, T alpha,
        Matrix<T, OrderA, Par> &A, Matrix<T, OrderB, Par> &B, T beta, Matrix<T, OrderC, Par> &C,
        Matrix<T, OrderC, Par> &result, T *buffer = nullptr) {
#pragma HLS INLINE
#ifndef __SYNTHESIS__
  assert((n % Par) == 0);
  assert((m % Par) == 0);
  assert((k % Par) == 0);
  assert(A.rows() == m);
  assert(A.cols() == k);
  assert(B.rows() == k);
  assert(B.cols() == n);
  assert(C.rows() == m);
  assert(C.cols() == n);
  assert(result.rows() == m);
  assert(result.cols() == n);
#endif

  if (OrderA != OrderB) {
    Matrix<T, OrderC, Par> AB(m, n);
    mm(m, n, k, alpha, A, B, AB, buffer);
    axpy(m, n, beta, C, AB, result);

#ifndef __SYNTHESIS__
    assert(A.empty());
    assert(B.empty());
    assert(C.empty());
    assert(AB.empty());
#endif
  } else {
// There shouldn't be any other option
#ifndef __SYNTHESIS__
    assert(("gemm with two matching major order inputs hasn't been implemented yet", false));
#endif
  }
#ifndef __SYNTHESIS__
  assert(("Matrix isn't empty", A.empty()));
  assert(("Matrix isn't empty", B.empty()));
  assert(("Matrix isn't empty", C.empty()));
  assert(("Matrix is empty", !result.empty()));
#endif
}
// TODO: Subtemplates for gemm, hemm, symm, gemmtr
// TODO: Specific implementations for the standard: cgemm, dgemm, sgemm, zgemm,
// sgemmtr, dgemmtr, cgemmtr, zgemmtr,
//       chemm, zhemm, csymm, zsymm, ssymm, dsymm
// TODO: Original specificaiton takes options to pre-transpose A and B, as well
// as conjugate transpose
// TODO: Allow customization of the order of C and result (I think they should
// remain the same to each other)

}  // namespace blas
}  // namespace dyfc

#endif  // DYFC_BLAS_MM_HPP
