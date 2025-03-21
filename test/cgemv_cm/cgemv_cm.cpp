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

#include "cgemv_cm.hpp"

#include "blas.hpp"

void cgemv_cm(ComplexFloat alpha, ComplexFloat A[dimN][dimM], ComplexFloat x[dimN],
              ComplexFloat beta, ComplexFloat y[dimN], ComplexFloat r[dimN]) {
#pragma HLS DATAFLOW

  // Load parameters into vectors and matrices. 2D arrays must be flattened before passing to
  // constructor
  dyfc::blas::Vector<ComplexFloat> x_v(x, dimN);
  dyfc::blas::Vector<ComplexFloat> y_v(y, dimM);
  dyfc::blas::Matrix<ComplexFloat, dyfc::blas::ColMajor> A_m(FLATTEN_MATRIX(A), dimM, dimN);
  dyfc::blas::Vector<ComplexFloat> r_v(dimM);

  // Call a templated version of the blas function being tested
  ComplexFloat buffer[dimM];
  dyfc::blas::mv<ComplexFloat, dyfc::blas::ColMajor>(dimM, dimN, alpha, A_m, x_v, beta, y_v,
                                                           r_v, buffer);

  // Write the result back to the output array
  r_v.to_memory(r);

  return;
}
