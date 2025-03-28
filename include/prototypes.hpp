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

#ifndef DYFC_BLAS_PROTOTYPES_HPP
#define DYFC_BLAS_PROTOTYPES_HPP

#include "complex.hpp"
#include "types.hpp"

namespace dyfc {
namespace blas {

// This is effectively a TODO list of all the functions that need to be
// implemented

/*
 * ===========================================================================
 * Prototypes for level 1 BLAS functions. Vector ops.
 * ===========================================================================
 */

/**
 * Computes a vector-scalar product and adds the result to a vector.
 *
 * r = alpha * x + y
 *
 * @tparam T The type of the elements in the vector. Supports any type with
 * defined arithmetic ops.
 * @tparam Par Number of elements retrieved in one read operation. Must be a
 * power of 2.
 *
 * @param[in]  n The length of the vectors x and y.
 * @param[in]  alpha The scalar to multiply the input vector x by.
 * @param[in]  x The input vector to multiply.
 * @param[in]  y The input vector to add to the result.
 * @param[out] result The output vector to write to.
 */
template <typename T, unsigned int Par>
void axpy(unsigned int n, T alpha, Vector<T, Par> &x, Vector<T, Par> &y, Vector<T, Par> &result);
// TODO: Specific implementations for the standard: caxpy, daxpy, saxpy, zaxpy

/**
 * Computes the product of a vector by a scalar.
 *
 * r = alpha * x
 *
 * @tparam T The type of the elements in the vector. Supports any type with
 * defined arithmetic ops.
 * @tparam Par Number of elements retrieved in one read operation. Must be a
 * power of 2.
 *
 * @param[in]  n The length of the vectors x and y.
 * @param[in]  alpha The scalar to multiply the input vector x by.
 * @param[in]  x The input vector to multiply.
 * @param[out] result The output vector to write to.
 */
template <typename T, unsigned int Par>
void scal(unsigned int n, T alpha, Vector<T, Par> &x, Vector<T, Par> &result);
// TODO: Specific implementations for the standard: cscal, csscal, dscal, sscal,
// zdscal, zscal

/**
 * The BLAS copy operation is nonsensical on streams. If you want to slice or
 * reshape a vector/matrix, please use the provided class methods.
 */
template <typename T, unsigned int Par>
void copy(Vector<T, Par> x, unsigned int incX, Vector<T, Par> result) {
  (void)x;
  (void)incX;
  (void)result;
  assert(("Copy operation not possible on streams", false));
}

/**
 * The BLAS swap operation is not possible on streams, so this function will
 * always assert false.
 */
template <typename T, unsigned int Par>
void swap(Vector<T, Par> &x, Vector<T, Par> &y) {
  (void)x;
  (void)y;
  assert(("Swap operation not possible on streams", false));
}

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
template <typename T, const unsigned int Par>
void dot(unsigned int n, Vector<T, Par> &x, Vector<T, Par> &y, T &result);
// Special implementation of dot where T is a complex number
void dotu();
// Special implementation of dotc where we take the conjugate of x before
// multiplying
void dotc();
// TODO: Specific implementations for the standard: cdotc, cdotu, ddot, dsdot,
// sdot, sdsdot, zdotc, zdotu

/**
 * Computes the Euclidean norm of a vector.
 *
 * r = ||x||
 *
 * @tparam T The type of the elements in the vector. Supports any type with
 * defined arithmetic ops.
 * @tparam Par Number of elements retrieved in one read operation. Must be a
 * power of 2.
 *
 * @param[in]  n The length of the vectors x and y.
 * @param[in]  x The input vector to compute the norm of.
 * @param[out] result The output value to write to.
 */
template <typename T, unsigned int Par>
void nrm2(unsigned int n, Vector<T, Par> &x, T &result);
// TODO: Specific implementations for the standard: dnrm2, dznrm2, scnrm2, snrm2

/**
 * Computes the sum of magnitudes of the vector elements.
 *
 * r = Σ |Re(x_i)| + |Im(x_i)|
 *
 * @tparam T The type of the elements in the vector. Supports any type with
 * defined arithmetic ops.
 * @tparam Par Number of elements retrieved in one read operation. Must be a
 * power of 2.
 *
 * @param[in]  n The length of the vectors x and y.
 * @param[in]  x The input vector to compute the sum of magnitudes of.
 * @param[out] result The output value to write to.
 */
template <typename T, unsigned int Par>
void asum(unsigned int n, Vector<T, Par> &x, T &result);
// TODO: Specific implementations for the standard: sasum, dasum scasum, dzasum

/**
 * Computes the prefix sum of a vector. Uses Hilles-Steele algorithm
 *
 * r_i = Σ x_i for i in [0, i]
 *
 * @tparam T The type of the elements in the vector. Supports any type with
 * defined arithmetic ops.
 * @tparam Par Number of elements retrieved in one read operation. Must be a
 * power of 2.
 *
 * @param[in]  n The length of the vectors x and y.
 * @param[in]  x The input vector to compute the prefix sum of.
 * @param[out] result The output vector to write to.
 */
template <typename T, unsigned int Par>
void asum(unsigned int n, Vector<T, Par> &x, Vector<T, Par> &result);

/**
 * Finds the index of the element with maximum absolute value.
 *
 * @tparam T The type of the elements in the vector. Supports any type with
 * defined arithmetic ops.
 * @tparam Par Number of elements retrieved in one read operation. Must be a
 * power of 2.
 *
 * @param[in]  n The length of the vectors x and y.
 * @param[in]  x The input vector to find the maximum element of.
 * @param[out] result The index that the maximum element was at
 */
template <typename T, unsigned int Par>
void amax(unsigned int n, Vector<T, Par> &x, unsigned int &result);
// TODO: Specific implementations for the standard: icamax, idamax, isamax,
// izamax

/**
 * Given the Cartesian coordinates (a, b) of a point, these routines return the
 * parameters c, s, r, and z associated with the Givens rotation. The parameters
 * c and s define a unitary matrix such that:
 *
 * [ c  s] [a] = [r]
 * [-s  c] [b] = [0]
 *
 * or, if using complex numbers
 *
 * [ c         s] [a] = [r]
 * [-conjg(s)  c] [b] = [0]
 *
 * The value r is non-negative.
 * c**2 + |s|**2 = 1
 *
 * @tparam T The type of the elements in the vector. Supports any type with
 * defined arithmetic ops.
 *
 * @param[in]  a The x coordinate of a point
 * @param[in]  b The y coordinate of a point
 * @param[out] c The cosine of the Givens rotation
 * @param[out] s The sine of the Givens rotation
 * @param[out] r The length of the vector <a, b>
 * @param[out] z A parameter that is equal to s if |a|>|b|, 1/c if c is not
 * zero, and 1 otherwise.
 */
template <typename T>
void rotg(T a, T b, T &c, T &s, T &r, T &z);
template <typename T>
void rotg(Complex<T> a, Complex<T> b, T &c, Complex<T> &s, Complex<T> &r);
// TODO: Make special cases for:
// 1. Real a, b, c, s, r, z
// 2. Complex a, b, c, s, r, z  <- This is not in the standard, but it's a
// useful extension
// 3. Real c. Complex a, b, s, r, z
// TODO: Specific implementations for the standard: crotg, drotg, srotg, zrotg

/**
 * Performs rotation of points in the plane.
 *
 * Given two complex vectors x and y, each vector element is replaced as follows
 *
 * NOTE: May need to insist that s and r are both complex, deviating from the
 * BLAS specification.
 *
 * xr = c * x + s * y
 * yr = c * y - conjg(s) * x
 *
 * @tparam T The type of the elements in the vector. Supports any type with
 * defined arithmetic ops.
 * @tparam Par Number of elements retrieved in one read operation. Must be a
 * power of 2.
 *
 * @param[in]  n The length of the vectors x and y.
 * @param[in]  x Set of x coordinates of points
 * @param[in]  y Set of y coordinates of points
 * @param[in]  c The cosine of the Givens rotation
 * @param[in]  s The sine of the Givens rotation
 * @param[out] xr The x coordinates of the rotated points
 * @param[out] yr The y coordinates of the rotated points
 */
template <typename T, unsigned int Par>
void rot(unsigned int n, Vector<T, Par> &x, Vector<T, Par> &y, T &c, T &s, Vector<T, Par> &xr,
         Vector<T, Par> &yr);
template <typename T, unsigned int Par>
void rot(unsigned int n, Vector<Complex<T>, Par> &x, Vector<Complex<T>, Par> &y, T &c,
         Complex<T> &s, Vector<Complex<T>, Par> &xr, Vector<Complex<T>, Par> &yr);
template <typename T, unsigned int Par>
void rot(unsigned int n, Vector<Complex<T>, Par> &x, Vector<Complex<T>, Par> &y, T &c, T &s,
         Vector<Complex<T>, Par> &xr, Vector<Complex<T>, Par> &yr);
// TODO: Make special cases for:
// 1. Real x, y, c, s
// 2. Complex x, y, c, s
// 3. Real x, y, c. Complex s.
// 4. Read c, s. Complex x,y
// TODO: Specific implementations for the standard: csrot, drot, srot, zrot,
// zdrot, crot

/**
 * Performs rotation of points in the plane. This is a modified version of rotg
 * that's only defined for real numbers.
 *
 * Given two real vectors x and y, each vector element is replaced as follows
 *
 * [ xr ] = H [ x1  sqrt(d1) ]
 * [ 0  ]     [ y1  sqrt(d2) ]
 *
 * Where H is a 2x2 matrix stored in row major order in the last 4 elements of
 * params. The first element of params is a flag element that is set as follows:
 *
 * params[0] = -1.0 => H = [ h11  h12 ]
 *                         [ h21  h22 ]
 *
 * params[0] =  0.0 => H = [ 1.0  h12 ]
 *                         [ h21  1.0 ]
 *
 * params[0] =  1.0 => H = [ h11  1.0 ]
 *                         [ -1.0 h22 ]
 *
 * params[0] = -2.0 => H = [ 1.0  0.0 ]
 *                         [ 0.0  1.0 ]
 *
 * In the last three cases, the matrix entries of 1.0, -1.0, and 0.0 are assumed
 * based on the value of flag and are not required to be set in the param
 * vector.
 *
 * @tparam T The type of the elements in the vector. Supports any type with
 * defined arithmetic ops.
 * @tparam Par Number of elements retrieved in one read operation. Must be a
 * power of 2.
 *
 * @param d1 The scaling factor for the x-coordinate of the input vector
 * @param d2 The scaling factor for the y-coordinate of the input vector
 * @param x1 The x-coordinate of the input vector
 * @param y1 The y-coordinate of the input vector
 * @param d1r The first diagonal element of the Givens rotation matrix
 * @param d2r The second diagonal element of the Givens rotation matrix
 * @param xr The x-coordinate of the rotated vector before scaling
 * @param params The parameters for the Givens rotation
 */
template <typename T, unsigned int Par>
void rotmg(T d1, T d2, T x1, T y1, T &d1r, T &d2r, T &xr, WideType<T, 5> &params);
// TODO: Add implementation for complex numbers, just to throw an unsupported
// warning
// TODO: Specific implementations for the standard: drotmg, srotmg

/*
 * Performs modified Givens rotation (rot) that only takes real numbers
 *
 * Given two real vectors x and y, each vector element is replaced as follows
 *
 * [ xr ] = H [ x1 ]
 * [ yr ]     [ y1 ]
 *
 * Where H is a 2x2 matrix stored in row major order in the last 4 elements of
 * params.
 *
 * @tparam T The type of the elements in the vector. Supports any type with
 * defined arithmetic ops.
 * @tparam Par Number of elements retrieved in one read operation. Must be a
 * power of 2.
 *
 * @param[in]  n The length of the vectors x and y.
 * @param[in]  x The x coordinates of a set of points to rotate
 * @param[in]  y The y coordinates of a set of points to rotate
 * @param[in]  H The Givens rotation matrix
 * @param[out] xr The x coordinates of the rotated points
 * @param[out] yr The y coordinates of the rotated points
 */
template <typename T, unsigned int Par>
void rotm(unsigned int n, Vector<T, Par> &x, Vector<T, Par> &y, WideType<T, 5> &H,
          Vector<T, Par> &xr, Vector<T, Par> &yr);
// TODO: Add implementation for complex numbers, just to throw an unsupported
// warning
// TODO: Specific implementations for the standard: drotm, srotm

/*
 * ===========================================================================
 * Prototypes for level 2 BLAS - Vector Matrix ops
 * ===========================================================================
 */

/**
 * Computes a matrix-vector product and adds the result to a vector.
 *
 * r = alpha * A * x + beta * y
 *
 * @tparam T The type of the elements in the matrix and vectors. Supports any
 * type with defined arithmetic ops.
 * @tparam Par Number of elements retrieved in one read operation. Must be a
 * power of 2.
 *
 * @param[in]  m The number of cols in the matrix A and the length of the vector y.
 *             Ignored if A is Triangular, Symmetric, or Hermitian, as these are square matrices.
 * @param[in]  n The number of rows in the matrix A and the length of the output vector r.
 * @param[in]  alpha The scalar to multiply the input matrix A by.
 * @param[in]  A The input matrix to multiply.
 * @param[in]  x The input vector to multiply.
 * @param[in]  beta The scalar to multiply the input vector y by.
 * @param[in]  y The input vector to add to the result.
 * @param[out] result The output vector to write to.
 */
template <typename T, unsigned int Par, MajorOrder Order = RowMajor>
void mv(unsigned int m, unsigned int n, T alpha, Matrix<T, Par, Order> &A, Vector<T, Par> &x,
        T beta, Vector<T, Par> &y, Vector<T, Par> &result);
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

/**
 * Computes a matrix-vector product.
 *
 * r = alpha * A * x
 *
 * @tparam T The type of the elements in the matrix and vectors. Supports any
 * type with defined arithmetic ops.
 * @tparam Par Number of elements retrieved in one read operation. Must be a
 * power of 2.
 *
 * @param[in]  m The number of cols in the matrix A and the length of the vector y.
 *            Ignored if A is Triangular, Symmetric, or Hermitian, as these are square matrices.
 * @param[in]  n The number of rows in the matrix A and the length of the output vector r.
 * @param[in]  alpha The scalar to multiply the input matrix A by.
 * @param[in]  A The input matrix to multiply.
 * @param[in]  x The input vector to multiply.
 * @param[in]  beta The scalar to multiply the input vector y by.
 * @param[in]  y The input vector to add to the result.
 * @param[out] result The output vector to write to.
 */
template <typename T, unsigned int Par, MajorOrder Order = RowMajor>
void mv(unsigned int m, unsigned int n, T alpha, Matrix<T, Par, Order> &A, Vector<T, Par> &x,
        Vector<T, Par> &result);
// TODO: Subtemplate for trmv, tbmv, tpmv
// TODO: Specific implementations for the standard: strmv, dtrmv, ctrmv, ztrmv,
// stbmv, dtbmv, ctbmv, ztbmv,
//       stpmv, dtpmv, ctpmv, ztpmv, ctpmv, ztpmv

/**
 * Computes a matrix-vector product for a triangular matrix and adds the result
 * to a vector.
 *
 * r = A * x
 *
 * @tparam T The type of the elements in the matrix and vectors. Supports any
 * type with defined arithmetic ops.
 * @tparam Par Number of elements retrieved in one read operation. Must be a
 * power of 2.
 *
 * @param[in]  n The number of rows/columns in the matrix A and the length of the vectors x and r.
 * @param[in]  A The input triangular matrix to multiply.
 * @param[in]  x The input vector to multiply.
 * @param[out] result The output vector to write to.
 */
template <typename T, unsigned int Par, MajorOrder Order = RowMajor, UpperLower UpLo = Upper>
void trmv(unsigned int n, TriangularMatrix<T, Par, Order, UpLo> &A, Vector<T, Par> &x,
          Vector<T, Par> &result);
// TODO: Subtemplates for trmv, tbmv, tpmv
// TODO: Specific implementations for the standard: ctrmv, dtrmv, strmv, ztrmv,
// ctbmv, dtbmv, stbmv, ztbmv, ctpmv, dtpmv, stpmv, ztpmv
// TODO: Original standard also takes options for if the matrix should be
// transposed or if the diagonal is a unit

/**
 * Solves a system of linear equations whose coefficients are a triangular
 * matrix.
 *
 * Finds r such that A * r = x
 *
 * Performs
 * r = A^-1 * x
 *
 * @tparam T The type of the elements in the matrix and vectors. Supports any
 * type with defined arithmetic ops.
 * @tparam Par Number of elements retrieved in one read operation. Must be a
 * power of 2.
 *
 * @param[in]  n The number of rows/columns in the matrix A and the length of the vectors x and r.
 * @param[in]  A The input triangular matrix to solve.
 * @param[in]  x The input vector to solve for.
 * @param[out] result The output vector to write to.
 */
template <typename T, unsigned int Par, MajorOrder Order = RowMajor, UpperLower UpLo = Upper>
void trsv(unsigned int n, TriangularMatrix<T, Par, Order, UpLo> &A, Vector<T, Par> &x,
          Vector<T, Par> &result);
// TODO: Subtemplates for trsv, tbsv, tpsv
// TODO: Specific implementations for the standard: ctrsv, dtrsv, strsv, ztrsv,
// ctbsv, dtbsv, stbsv, ztbsv,
//       ctpsv, dtpsv, stpsv, ztpsv
// TODO: Original standard also takes options for if the matrix should be
// transposed or if the diagonal is a unit

/**
 * Performs a rank-1 update of a general matrix
 *
 * r = alpha * x * y^T + A
 *
 * @tparam T The type of the elements in the matrix and vectors. Supports any
 * type with defined arithmetic ops.
 * @tparam Par Number of elements retrieved in one read operation. Must be a
 * power of 2.
 *
 * @param[in]  m The number of rows in the matrix A and the length of the input vector x.
 * @param[in]  n The number of columns in the matrix A and the length of the input vector y.
 * @param[in]  alpha The scalar to multiply the outer product of x and y by.
 * @param[in]  x The input vector to multiply.
 * @param[in]  y The input vector to multiply.
 * @param[in]  A The input matrix to add to.
 * @param[out] result The output matrix to write to.
 */
template <typename T, unsigned int Par>
void ger(unsigned int m, unsigned int n, T alpha, Vector<T, Par> &x, Vector<T, Par> &y,
         Matrix<T, Par> &A, Matrix<T, Par> &result);
// TODO: Subtemplates for syr, her, spr, hpr
// TODO: Specific implementations for the standard: cgeru, zgeru, dger, sger,
// ssyr, dsyr, cher, zher, sspr, dspr, chpr, zhpr

/**
 * Performs a rank-1 update of a general matrix with complex numbers, using the
 * conjugate transpose of y
 *
 * r = alpha * x * conj(y)^T + A
 *
 * @tparam T The type of the elements in the matrix and vectors. Supports any
 * type with defined arithmetic ops.
 * @tparam Par Number of elements retrieved in one read operation. Must be a
 * power of 2.
 * @tparam Order The major order of the matrix. Can be either RowMajor or
 * ColMajor.
 *
 * @param[in]  m The number of rows in the matrix A and the length of the input vector x.
 * @param[in]  n The number of columns in the matrix A and the length of the input vector y.
 * @param[in]  alpha The scalar to multiply the outer product of x and y by.
 * @param[in]  x The input vector to multiply.
 * @param[in]  y The input vector to multiply.
 * @param[in]  A The input matrix to add to.
 * @param[out] result The output matrix to write to.
 */
template <typename T, unsigned int Par, MajorOrder Order = RowMajor>
void gerc(unsigned int m, unsigned int n, T alpha, Vector<Complex<T>, Par> &x,
          Vector<Complex<T>, Par> &y, Matrix<Complex<T>, Par, Order> &A,
          Matrix<Complex<T>, Par, Order> &result);
// TODO: Specific implementations for the standard: cgerc, zgerc

/**
 * Performs a rank-2 update of a symmetric matrix
 *
 * r = alpha * x * y^T + alpha * y * x^T + A
 *
 * @tparam T The type of the elements in the matrix and vectors. Supports any
 * type with defined arithmetic ops.
 * @tparam Par Number of elements retrieved in one read operation. Must be a
 * power of 2.
 * @tparam Order The major order of the matrix. Can be either RowMajor or
 * ColMajor.
 * @tparam UpLo Whether the matrix is upper or lower triangular.
 *
 * @param[in]  n The number of rows/cols in the matrix A and the length of the input vector x and y.
 * @param[in]  alpha The scalar to multiply the outer product of x and y by.
 * @param[in]  x The input vector to multiply.
 * @param[in]  y The input vector to multiply.
 * @param[in]  A The input matrix to add to.
 * @param[out] result The output matrix to write to.
 */
template <typename T, unsigned int Par, MajorOrder Order = RowMajor, UpperLower UpLo = Upper>
void syr2(unsigned int n, T alpha, Vector<T, Par> &x, Vector<T, Par> &y,
          SymmetricMatrix<T, Par, Order, UpLo> &A, SymmetricMatrix<T, Par, Order, UpLo> &result);
// TODO: Specific implementations for the standard: dsyr2, ssyr2, sspr2, dspr2

/**
 * Peforms a rank-2 update of a Hermitean matrix
 *
 * r = alpha * x * conj(y)^T + y * conj(alpha * x)^T + A
 *
 * @tparam T The type of the elements in the matrix and vectors. Supports any
 * type with defined arithmetic ops.
 * @tparam Par Number of elements retrieved in one read operation. Must be a
 * power of 2.
 * @tparam Order The major order of the matrix. Can be either RowMajor or
 * ColMajor.
 * @tparam UpLo Whether the matrix is upper or lower triangular.
 *
 * @param[in]  n The number of rows/cols in the matrix A and the length of the input vector x and y.
 * @param[in]  alpha The scalar to multiply the outer product of x and y by.
 * @param[in]  x The input vector to multiply.
 * @param[in]  y The input vector to multiply.
 * @param[in]  A The input matrix to add to.
 * @param[out] result The output matrix to write to.
 */
template <typename T, unsigned int Par, MajorOrder Order = RowMajor, UpperLower UpLo = Upper>
void her2(unsigned int n, T alpha, Vector<Complex<T>, Par> &x, Vector<Complex<T>, Par> &y,
          HermitianMatrix<Complex<T>, Par, Order, UpLo> &A,
          HermitianMatrix<Complex<T>, Par, Order, UpLo> &result);
// TODO: Specific implementations for the standard: cher2, zher2, chpr2, zhpr2

/*
 * ===========================================================================
 * Prototypes for level 3 BLAS - Matrix Matrix ops
 * ===========================================================================
 */

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
template <typename T, unsigned int Par, MajorOrder OrderA = RowMajor, MajorOrder OrderB = ColMajor,
          UpperLower UpLo = Upper>
void mm(unsigned int m, unsigned int n, unsigned int k, T alpha, Matrix<T, Par, OrderA> &A,
        Matrix<T, Par, OrderB> &B, T beta, Matrix<T, Par, OrderA> &C,
        Matrix<T, Par, OrderA> &result);
// TODO: Subtemplates for gemm, hemm, symm, gemmtr
// TODO: Specific implementations for the standard: cgemm, dgemm, sgemm, zgemm,
// sgemmtr, dgemmtr, cgemmtr, zgemmtr,
//       chemm, zhemm, csymm, zsymm, ssymm, dsymm
// TODO: Original specificaiton takes options to pre-transpose A and B, as well
// as conjugate transpose
// TODO: Allow customization of the order of C and result (I think they should
// remain the same to each other)

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
template <typename T, unsigned int Par, MajorOrder OrderA = RowMajor, MajorOrder OrderB = ColMajor,
          UpperLower UpLo = Upper>
void mm(unsigned int m, unsigned int n, unsigned int k, T alpha, Matrix<T, Par, OrderA> &A,
        Matrix<T, Par, OrderB> &B, Matrix<T, Par, OrderA> &result);
// TODO: Subtemplates for gemm, hemm, symm, trmm
// TODO: Specific implementations for the standard: cgemm, dgemm, sgemm, zgemm,
// strmm, dtrmm, ctrmm, ztrmm, chemm, zhemm, csymm, zsymm, ssymm, dsymm
// TODO: Original specificaiton takes options to pre-transpose A and B

/**
 * Solves a triangular matrix equation
 *
 * r = alpha * A^-1 * B
 *
 * @tparam T The type of the elements in the matrix and vectors. Supports any
 * type with defined arithmetic ops.
 * @tparam Par Number of elements retrieved in one read operation. Must be a
 * power of 2.
 * @tparam Order The major order of the matrix. Can be either RowMajor or
 * ColMajor.
 * @tparam UpLo Whether the matrix is upper or lower triangular.
 *
 * @param[in]  m The number of rows in the matrices A, B, and the output matrix result.
 * @param[in]  n The number of columns in the matrix B and the output matrix result.
 * @param[in]  alpha The scalar to multiply the input matrix A by.
 * @param[in]  A The input triangular matrix to solve. Dimensions are m x m.
 * @param[in]  B The input matrix to solve for. Dimensions are m x n.
 * @param[out] result The output matrix to write to. Dimensions are m x n.
 */
template <typename T, unsigned int Par, MajorOrder Order = RowMajor, UpperLower UpLo = Upper>
void trsm(unsigned int m, unsigned int n, T alpha, TriangularMatrix<T, Par, Order, UpLo> &A,
          Matrix<T, Par> &B, Matrix<T, Par> &result);
// TODO: Specific implementations for the standard: ctrsm, dtrsm, strsm, ztrsm
// TODO: Original specificaiton takes options to pre-transpose A and B, as well
// as take their conjugate transpose

/**
 * Performs a symmetric rank-k update of a matrix
 *
 * r = alpha * A * A^T + beta * C
 *
 * @tparam T The type of the elements in the matrix and vectors. Supports any
 * type with defined arithmetic ops.
 * @tparam Par Number of elements retrieved in one read operation. Must be a
 * power of 2.
 * @tparam Order The major order of the matrix. Can be either RowMajor or
 * ColMajor.
 * @tparam UpLo Whether the matrix C is upper or lower triangular.
 *
 * @param[in]  n The number of rows in the matrices A, C, and the output matrix result.
 * @param[in]  k The number of columns in the matrix A.
 * @param[in]  alpha The scalar to multiply the outer product of A*A^T by.
 * @param[in]  A The input matrix to multiply.
 * @param[in]  beta The scalar to multiply the input matrix C by.
 * @param[in]  C The input matrix to add to. Is square and symmetric.
 * @param[out] result The output matrix to write to.
 */
template <typename T, unsigned int Par, MajorOrder Order = RowMajor, UpperLower UpLo = Upper>
void syrk(unsigned int n, unsigned int k, T alpha, Matrix<T, Par, Order> &A, T beta,
          SymmetricMatrix<T, Par, Order, UpLo> &C, SymmetricMatrix<T, Par, Order, UpLo> &result);
// TODO: Specific implementations for the standard: csyrk, dsyrk, ssyrk, zsyrk

/**
 * Performs a Hermitian rank-k update of a matrix
 *
 * r = alpha * A * conj(A)^T + beta * C
 *
 * @tparam T The type of the elements in the matrix and vectors. Supports any
 * type with defined arithmetic ops.
 * @tparam Par Number of elements retrieved in one read operation. Must be a
 * power of 2.
 * @tparam Order The major order of the matrix. Can be either RowMajor or
 * ColMajor.
 * @tparam UpLo Whether the matrix C is upper or lower triangular.
 *
 * @param[in]  n The number of rows in the matrices A, C, and the output matrix result.
 * @param[in]  k The number of columns in the matrix A.
 * @param[in]  alpha The scalar to multiply the outer product of A*conj(A)^T by.
 * @param[in]  A The input matrix to multiply.
 * @param[in]  beta The scalar to multiply the input matrix C by.
 * @param[in]  C The input matrix to add to. Is square and Hermitian.
 * @param[out] result The output matrix to write to.
 */
template <typename T, unsigned int Par, MajorOrder Order = RowMajor, UpperLower UpLo = Upper>
void herk(unsigned int n, unsigned int k, T alpha, Matrix<Complex<T>, Par, Order> &A, T beta,
          HermitianMatrix<Complex<T>, Par, Order, UpLo> &C,
          HermitianMatrix<Complex<T>, Par, Order, UpLo> &result);
// TODO: Specific implementations for the standard: cherk, zherk

/**
 * Performs a symmetric rank-2k update of a matrix
 *
 * r = alpha * A * B^T + alpha * B * A^T + beta * C
 *
 * @tparam T The type of the elements in the matrix and vectors. Supports any
 * type with defined arithmetic ops.
 * @tparam Par Number of elements retrieved in one read operation. Must be a
 * power of 2.
 * @tparam Order The major order of the matrix. Can be either RowMajor or
 * ColMajor.
 * @tparam UpLo Whether the matrix C is upper or lower triangular.
 *
 * @param[in]  n The number of rows in the matrices A, B, C, and the output matrix result.
 * @param[in]  k The number of columns in the matrices A and B.
 * @param[in]  alpha The scalar to multiply the outer product of A*B^T and B*A^T
 * by.
 * @param[in]  A The input matrix to multiply.
 * @param[in]  B The input matrix to multiply.
 * @param[in]  beta The scalar to multiply the input matrix C by.
 * @param[in]  C The input matrix to add to. Is square and symmetric.
 * @param[out] result The output matrix to write to.
 */
template <typename T, unsigned int Par, MajorOrder Order = RowMajor, UpperLower UpLo = Upper>
void syr2k(unsigned int n, unsigned int k, T alpha, Matrix<T, Par, Order> &A,
           Matrix<T, Par, Order> &B, T beta, SymmetricMatrix<T, Par, Order, UpLo> &C,
           SymmetricMatrix<T, Par, Order, UpLo> &result);
// TODO: Specific implementations for the standard: csyr2k, dsyr2k, ssyr2k,
// zsyr2k

/**
 * Performs a Hermitian rank-2k update of a matrix
 *
 * r = alpha * A * conj(B)^T + alpha * B * conj(A)^T + beta * C
 *
 * @tparam T The type of the elements in the matrix and vectors. Supports any
 * type with defined arithmetic ops.
 * @tparam Par Number of elements retrieved in one read operation. Must be a
 * power of 2.
 * @tparam Order The major order of the matrix. Can be either RowMajor or
 * ColMajor.
 * @tparam UpLo Whether the matrix C is upper or lower triangular.
 *
 * @param[in]  n The number of rows in the matrices A, B, C, and the output matrix result.
 * @param[in]  k The number of columns in the matrices A and B.
 * @param[in]  alpha The scalar to multiply the outer product of A*conj(B)^T and
 * B*conj(A)^T by.
 * @param[in]  A The input matrix to multiply.
 * @param[in]  B The input matrix to multiply.
 * @param[in]  beta The scalar to multiply the input matrix C by.
 * @param[in]  C The input matrix to add to. Is square and Hermitian.
 * @param[out] result The output matrix to write to.
 */
template <typename T, unsigned int Par, MajorOrder Order = RowMajor, UpperLower UpLo = Upper>
void her2k(unsigned int n, unsigned int k, T alpha, Matrix<Complex<T>, Par, Order> &A,
           Matrix<Complex<T>, Par, Order> &B, T beta,
           HermitianMatrix<Complex<T>, Par, Order, UpLo> &C,
           HermitianMatrix<Complex<T>, Par, Order, UpLo> &result);
// TODO: Specific implementations for the standard: cher2k, zher2k

/**
 * Transposes a column-major matrix into a row-major matrix, or vice versa.
 * Doesn't move any values, just changes the type.
 */
template <typename T, unsigned int Par>
Matrix<T, Par, RowMajor> transpose(Matrix<T, Par, ColMajor> &A);
template <typename T, unsigned int Par>
Matrix<T, Par, ColMajor> transpose(Matrix<T, Par, RowMajor> &A);

/**
 * Transposes a column-major matrix into a row-major matrix, or vice versa. This
 * requires rearranging the data in memory, so it's a more expensive operation.
 */
template <typename T, unsigned int Par>
Matrix<T, Par, ColMajor> transpose(Matrix<T, Par, ColMajor> &A);
template <typename T, unsigned int Par>
Matrix<T, Par, RowMajor> transpose(Matrix<T, Par, RowMajor> &A);

}  // namespace blas
}  // namespace dyfc

#endif  // DYFC_BLAS_PROTOTYPES_HPP
