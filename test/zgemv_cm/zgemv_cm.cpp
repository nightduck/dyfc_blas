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

#include "zgemv_cm.hpp"

#include "blas.hpp"

void zgemv_cm(ComplexDouble alpha, ComplexDouble A[dimN][dimM], ComplexDouble x[dimN],
              ComplexDouble beta, ComplexDouble y[dimN], ComplexDouble r[dimN]) {
#pragma HLS DATAFLOW

  // Load parameters into vectors and matrices. 2D arrays must be flattened before passing to
  // constructor
  dyfc::blas::Vector<ComplexDouble> x_v(x, dimN);
  dyfc::blas::Vector<ComplexDouble> y_v(y, dimM);
  dyfc::blas::Matrix<ComplexDouble, dyfc::blas::ColMajor> A_m(FLATTEN_MATRIX(A), dimM, dimN);
  dyfc::blas::Vector<ComplexDouble> r_v(dimM);

  // Call a templated version of the blas function being tested
  ComplexDouble buffer[dimM];
  dyfc::blas::mv<ComplexDouble, dyfc::blas::ColMajor>(dimM, dimN, alpha, A_m, x_v, beta, y_v, r_v,
                                                      buffer);

  // Write the result back to the output array
  r_v.to_memory(r);

  return;
}
