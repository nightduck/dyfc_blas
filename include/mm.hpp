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

#include "complex.hpp"
#include "prefixsum.hpp"
#include "types.hpp"

namespace dyfc {
namespace blas {

/**
 * Performs matrix multiplication
 *
 * r = alpha * A * B
 *
 * @tparam T The type of the elements in the matrix and vectors. Supports any
 * type with defined arithmetic ops.
 * @tparam Par Number of elements retrieved in one read operation. Must be a
 * power of 2.
 * @tparam OrderA The major order of the matrix A. Can be either RowMajor or
 * ColMajor.
 * @tparam OrderB The major order of the matrix B. Can be either RowMajor or
 * ColMajor.
 * @tparam UpLo Whether the matrix is upper or lower triangular. This is ignored
 * for general matrices
 *
 * @param[in]  m The number of rows in the matrix A and the output matrix result.
 * @param[in]  n The number of columns in the matrix B and the output matrix result.
 * @param[in]  k The number of columns in the matrix A and the number of rows in the matrix B.
 * @param[in]  alpha The scalar to multiply the input matrix A by.
 * @param[in]  A The input matrix to multiply.
 * @param[in]  B The input matrix to multiply.
 * @param[out] result The output matrix to write to.
 */
template <typename T, unsigned int Par,
          MajorOrder OrderA = RowMajor, MajorOrder OrderB = ColMajor, UpperLower UpLo = Upper>
void mm(unsigned int m, unsigned int n, unsigned int k, T alpha, Matrix<T, Par, OrderA> &A, Matrix<T, Par, OrderB> &B,
        Matrix<T, Par, OrderA> &result) {
#pragma HLS INLINE
#ifndef __SYNTHESIS__
  assert((n % Par) == 0);
  assert((m % Par) == 0);
  assert(A.cols() == k);
  assert(A.rows() == m);
  assert(B.rows() == k);
  assert(B.cols() == n);
  assert(result.rows() == m);
  assert(result.cols() == n);
#endif
  
  if (OrderA == RowMajor && OrderB == ColMajor) {
    T r = T(0);
    hls::stream<WideType<T, Par>> A_row_stream;
    hls::stream<WideType<T, Par>> B_repeat_stream;
    for(int i = 0; i < m; i++) {
      for(int j = 0; j < n; j ++) {
        for(int p = 0; p < k; p+=Par) {
          WideType<T, Par> A_val;
          WideType<T, Par> B_val;
          if (p == 0) {
            A_val = A.read();
            A_row_stream.write(A_val);
          } else if (p + Par > k) {
            A_val = A_row_stream.read();
          } else {
            A_val = A_row_stream.read();
            A_row_stream.write(A_val);
          }
          if (i == 0) {
            B_val = B.read();
            B_repeat_stream.write(B_val);
          } else if (i == m - 1) {
            B_val = B_repeat_stream.read();
          } else {
            B_val = B_repeat_stream.read();
            B_repeat_stream.write(B_val);
          }

          WideType<T, Par> r_val;
          WideType<T, Par> rsum_val = T(0);
          for(int s = 0; s < Par; s++) {
#pragma HLS UNROLL
            r_val[s] = A_val[s] * B_val[s];
          }
          prefixsum<T, Par>(r_val, rsum_val, r);
          r = rsum_val[Par - 1];
          if (p + Par >= k) {
            result.write(r*alpha);
            r = T(0);
          }
        }
      }
    }
  } else if (OrderA == ColMajor && OrderB == RowMajor) {
  
  } else if (OrderA == RowMajor && OrderB == RowMajor) {
    // There shouldn't be any other option
    #ifndef __SYNTHESIS__
    assert(("gemm with two row major order inputs hasn't been implemented yet", false));
    #endif
  } else if (OrderA == ColMajor && OrderB == ColMajor) {
    // There shouldn't be any other option
    #ifndef __SYNTHESIS__
      assert(("gemm with two column major order inputs hasn't been implemented yet", false));
    #endif
  } else {
    // There shouldn't be any other option
    #ifndef __SYNTHESIS__
      assert(("Invalid MajorOrder option (this shouldn't be possible, wtf did you do?)", false));
    #endif
  }
}
// TODO: Subtemplates for gemm, hemm, symm, trmm
// TODO: Specific implementations for the standard: cgemm, dgemm, sgemm, zgemm,
// strmm, dtrmm, ctrmm, ztrmm, chemm, zhemm, csymm, zsymm, ssymm, dsymm
// TODO: Original specificaiton takes options to pre-transpose A and B

/**
 * Performs matrix multiplication
 *
 * r = alpha * A * B + beta * C
 *
 * @tparam T The type of the elements in the matrix and vectors. Supports any
 * type with defined arithmetic ops.
 * @tparam Par Number of elements retrieved in one read operation. Must be a
 * power of 2.
 * @tparam OrderA The major order of the matrix A. Can be either RowMajor or
 * ColMajor.
 * @tparam OrderB The major order of the matrix B. Can be either RowMajor or
 * ColMajor.
 * @tparam UpLo Whether the matrix is upper or lower triangular. This is ignored
 * for general matrices
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
template <typename T, unsigned int Par,
          MajorOrder OrderA = RowMajor, MajorOrder OrderB = ColMajor, UpperLower UpLo=Upper>
void mm(unsigned int m, unsigned int n, unsigned int k, T alpha, Matrix<T, Par, OrderA> &A, Matrix<T, Par, OrderB> &B, T beta,
        Matrix<T, Par, OrderA> &C, Matrix<T, Par, OrderA> &result) {
#pragma HLS INLINE
#ifndef __SYNTHESIS__
  assert((n % Par) == 0);
  assert((m % Par) == 0);
  assert(A.cols() == k);
  assert(A.rows() == m);
  assert(B.rows() == k);
  assert(B.cols() == n);
  assert(C.rows() == m);
  assert(C.cols() == n);
  assert(result.rows() == m);
  assert(result.cols() == n);
#endif

  if (OrderA != OrderB) {
    Matrix<T, Par, OrderA> AB(m, n);
    mm(m, n, k, alpha, A, B, AB);
    axpy(m, n, beta, C, AB, result);
  } else {
    // There shouldn't be any other option
    #ifndef __SYNTHESIS__
    assert(("gemm with two matching major order inputs hasn't been implemented yet", false));
    #endif
  }
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
