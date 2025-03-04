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

#include "transpose.hpp"

#include "blas.hpp"

void transpose_krnl(double A[dimN][dimM], double ArTc[dimN][dimM], double ArTr[dimM][dimN],
               double AcTr[dimN][dimM], double AcTc[dimM][dimN]) {
  // Load parameters into vectors and matrices. 2D arrays must be flattened before passing to
  // constructor
  // dyfc::blas::Matrix<double, Par, dyfc::blas::RowMajor> Ar_m1(FLATTEN_MATRIX(A), dimN, dimM);
  dyfc::blas::Matrix<double, dyfc::blas::RowMajor> Ar_m2(FLATTEN_MATRIX(A), dimN, dimM);
  dyfc::blas::Matrix<double, dyfc::blas::ColMajor> Ac_m1(FLATTEN_MATRIX(A), dimM, dimN);
  // dyfc::blas::Matrix<double, Par, dyfc::blas::ColMajor> Ac_m2(FLATTEN_MATRIX(A), dimM, dimN);

  // dyfc::blas::Matrix<double, Par, dyfc::blas::ColMajor> ArT_m1(dimM, dimN);
  dyfc::blas::Matrix<double, dyfc::blas::ColMajor> ArT_m2(dimM, dimN);
  dyfc::blas::Matrix<double, dyfc::blas::RowMajor> AcT_m1(dimN, dimM);
  // dyfc::blas::Matrix<double, Par, dyfc::blas::RowMajor> AcT_m2(dimN, dimM);

  // Perform the transpose
  // dyfc::blas::transpose(Ar_m1, ArT_m1);
  dyfc::blas::transpose(Ar_m2, ArT_m2);
  dyfc::blas::transpose(Ac_m1, AcT_m1);
  // dyfc::blas::transpose(Ac_m2, AcT_m2);

  // Write the result back to the output array
  // ArT_m1.to_memory<dyfc::blas::RowMajor>(FLATTEN_MATRIX(ArTr));
  ArT_m2.to_memory(FLATTEN_MATRIX(ArTc));
  AcT_m1.to_memory(FLATTEN_MATRIX(AcTr));
  // AcT_m2.to_memory<dyfc::blas::ColMajor>(FLATTEN_MATRIX(AcTc));

  return;
}
