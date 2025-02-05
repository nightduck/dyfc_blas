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

#ifndef DYFC_BLAS_TEST_EXAMPLE_HPP
#define DYFC_BLAS_TEST_EXAMPLE_HPP

// The size of the problem should be specified in preprocessor directives that can be overriden by
// the compile command. Follow the convention
#ifndef dimN
#define dimN 128
#endif
#ifndef dimM
#define dimM 128
#endif
#ifndef dimK
#define dimK 128
#endif

// Each test should target a particular example instantiation of a blas function, and the name
// should be annotated according to BLAS convention to reflect that.
// EG zaxpy is the axpy operation that takes double precision complex numbers.
// Illustrated here is the fictitious example function implemented with double precision real
// numbers and a column-major matrix.
void dexample_cm(double alpha, double x[dimN], double A[dimN][dimN], double r[dimN]);

#endif  // DYFC_BLAS_TEST_EXAMPLE_HPP
