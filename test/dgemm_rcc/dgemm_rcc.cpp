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

#include "dgemm_rcc.hpp"

#include "blas.hpp"

void dgemm_rcc(double alpha, double A[dimM][dimK], double B[dimN][dimK], double beta, double C[dimN][dimM], double r[dimN][dimM]) {
#pragma HLS DATAFLOW

  // Load parameters into vectors and matrices. 2D arrays must be flattened before passing to
  // constructor
  dyfc::blas::Matrix<double, dyfc::blas::RowMajor> A_m(FLATTEN_MATRIX(A), dimM, dimK);
  dyfc::blas::Matrix<double, dyfc::blas::ColMajor> B_m(FLATTEN_MATRIX(B), dimK, dimN);
  dyfc::blas::Matrix<double, dyfc::blas::ColMajor> C_m(FLATTEN_MATRIX(C), dimM, dimN);
  dyfc::blas::Matrix<double, dyfc::blas::ColMajor> r_m(dimM, dimN);

  // Call a templated version of the blas function being tested
  dyfc::blas::mm<double, dyfc::blas::RowMajor, dyfc::blas::ColMajor, dyfc::blas::ColMajor>(dimM, dimN, dimK, alpha, A_m, B_m, beta, C_m, r_m);

  // Write the result back to the output array
  r_m.to_memory(FLATTEN_MATRIX(r));

  return;
}
