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

#include "ddot.hpp"

#include "blas.hpp"

// Test both the basic asum and the prefix sum calls
void ddot(double x[dimN], double y[dimN], double &r) {
  // Suggested parallelism level: 4096 / 8 / sizeof(type)
  // EG: 64 for doubles, 128 for floats, 32 for double precision complex
  const int Par = 4096 / 8 / sizeof(double);

  // Load parameters into vectors and matrices. 2D arrays must be flattened before passing to
  // constructor
  dyfc::blas::Vector<double, Par> x_v(x, dimN);
  dyfc::blas::Vector<double, Par> y_v(y, dimN);

  // Call a templated version of the blas function being tested
  dyfc::blas::dot<double, Par>(dimN, x_v, y_v, r);

  return;
}
