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

// TODO: Explore requiring intermediate buffer depending on the permutation of matrix orders.
// EG RowMajor * ColMajor can be done in stream, but RowMajor * RowMajor requires a buffer of N (or
// M depending on the order of the output matrix), and ColMajor * RowMajor requires a buffer of M*N.
/**
 * Performs matrix multiplication
 *
 * r = alpha * A * B
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
 * @param[out] result The output matrix to write to.
 */
template <typename T, const MajorOrder OrderA = RowMajor, const MajorOrder OrderB = ColMajor,
          const MajorOrder OrderC = RowMajor, const UpperLower UpLo = Upper,
          const unsigned int Par = MAX_BITWIDTH / 8 / sizeof(T)>
void mm(unsigned int m, unsigned int n, unsigned int k, T alpha, Matrix<T, OrderA, Par> &A,
        Matrix<T, OrderB, Par> &B, Matrix<T, OrderA, Par> &result, T* buffer = nullptr) {
#pragma HLS INLINE
#ifndef __SYNTHESIS__
  assert((n % Par) == 0);
  assert((m % Par) == 0);
  assert((k % Par) == 0);
  assert(A.cols() == k);
  assert(A.rows() == m);
  assert(B.rows() == k);
  assert(B.cols() == n);
  assert(result.rows() == m);
  assert(result.cols() == n);
  assert(("This matrix is a pure stream and only accepts one reader", A.read_lock()));
  assert(("This matrix is a pure stream and only accepts one reader", B.read_lock()));
  assert(("This matrix only accepts one writer", result.write_lock()));
#endif

  typename Matrix<T, OrderA, Par>::StreamType A_stream;
  typename Matrix<T, OrderB, Par>::StreamType B_stream;
  if (OrderA == RowMajor && OrderB == ColMajor) {
#pragma HLS DATAFLOW
    A.read(A_stream, false, false, B.cols(), 1, 1, 1);
    B.read(B_stream, false, false, 1, 1, 1, A.rows());

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
  } else if (OrderA == ColMajor && OrderB == RowMajor) {
#ifndef __SYNTHESIS__
    assert(("gemm with two row major order inputs hasn't been implemented yet", false));
#endif
  } else if (OrderA == RowMajor && OrderB == RowMajor) {
#ifndef __SYNTHESIS__
    assert(("gemm with two row major order inputs hasn't been implemented yet", false));
#endif
  } else if (OrderA == ColMajor && OrderB == ColMajor) {
#ifndef __SYNTHESIS__
    assert(("gemm with two column major order inputs hasn't been implemented yet", false));
#endif
  } else {
// There shouldn't be any other option
#ifndef __SYNTHESIS__
    assert(("Invalid MajorOrder option (this shouldn't be possible, wtf did you do?)", false));
#endif
  }
#ifndef __SYNTHESIS__
  assert(("Matrix A isn't empty", A.empty()));
  assert(("Matrix isn't empty", B.empty()));
  assert(("A_stream isn't empty", A_stream.empty()));
  assert(("B_stream isn't empty", B_stream.empty()));
  assert(("Matrix result is unexpected size", result.size() == m * n / Par));
#endif
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
void mm(unsigned int m, unsigned int n, unsigned int k, T alpha, Matrix<T, OrderA, Par> &A,
        Matrix<T, OrderB, Par> &B, T beta, Matrix<T, OrderC, Par> &C,
        Matrix<T, OrderC, Par> &result, T* buffer = nullptr) { {
#pragma HLS INLINE
#ifndef __SYNTHESIS__
  assert((n % Par) == 0);
  assert((m % Par) == 0);
  assert((k % Par) == 0);
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
    Matrix<T, OrderA, Par> AB(m, n);
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
