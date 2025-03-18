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

#include "mat_inv.hpp"

#include "blas.hpp"

void mat_inv(double x[dimN][dimN], double r[dimN][dimN]) {
#pragma HLS DATAFLOW
    
  // Load parameters into vectors and matrices. 2D arrays must be flattened before passing to
  // constructor
  dyfc::blas::Matrix<double> x_m(FLATTEN_MATRIX(x), dimN, dimN);
  dyfc::blas::Matrix<double> r_m(dimN, dimN);
  double buffer[dimN][2*dimN];

  // Call a templated version of the blas function being tested
  bool result = x_m.invert(r_m, FLATTEN_MATRIX(buffer));
  #ifndef __SYNTHESIS__
  assert(("Matrix inversion failed", result == true));
  #endif

  // Write the result back to the output array
  r_m.to_memory(FLATTEN_MATRIX(r));

  return;
}
