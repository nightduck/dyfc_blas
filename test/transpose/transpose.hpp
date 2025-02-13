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

#ifndef DYFC_BLAS_TEST_TRANSPOSE_HPP
#define DYFC_BLAS_TEST_TRANSPOSE_HPP

#ifndef dimN
#define dimN 256
#endif
#ifndef dimM
#define dimM 512
#endif

void transpose_krnl(double A[dimN][dimM], double ArTc[dimN][dimM], double ArTr[dimM][dimN],
               double AcTr[dimN][dimM], double AcTc[dimM][dimN]);

#endif  // DYFC_BLAS_TEST_TRANSPOSE_HPP
