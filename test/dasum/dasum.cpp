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

#include "dasum.hpp"

#include "blas.hpp"

// Test both the basic asum and the prefix sum calls
void dasum(double x[dimN], double &r) {
  // Load parameters into vectors and matrices. 2D arrays must be flattened before passing to
  // constructor
  dyfc::blas::Vector<double> x_v(x, dimN);

  // Call a templated version of the blas function being tested
  dyfc::blas::asum<double>(dimN, x_v, r);

  return;
}
