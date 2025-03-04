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

#include "saxpy.hpp"

#include "blas.hpp"

void saxpy(float alpha, float x[dimN], float y[dimN], float r[dimN]) {
  // Load parameters into vectors and matrices. 2D arrays must be flattened before passing to
  // constructor
  dyfc::blas::Vector<float> x_v(x, dimN);
  dyfc::blas::Vector<float> y_v(y, dimN);
  dyfc::blas::Vector<float> r_v(dimN);

  // Call a templated version of the blas function being tested
  dyfc::blas::axpy<float>(dimN, alpha, x_v, y_v, r_v);

  // Write the result back to the output array
  r_v.write(r);

  return;
}
