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

#ifndef DYFC_BLAS_TEST_ZGEMV_RM_HPP
#define DYFC_BLAS_TEST_ZGEMV_RM_HPP

#include "complex.hpp"

#ifndef dimN
#define dimN 256
#endif
#ifndef dimM
#define dimM 512
#endif

// These are reference numbers for the benchmark, commenting them out will have no effect
#define dimNSweepMin 32
#define dimNSweepMax 4096
#define dimMSweepMin 32
#define dimMSweepMax 4096

using ComplexDouble = dyfc::blas::Complex<double>;

void zgemv_rm(ComplexDouble alpha, ComplexDouble A[dimM][dimN], ComplexDouble x[dimN],
              ComplexDouble beta, ComplexDouble y[dimN], ComplexDouble r[dimN]);

#endif  // DYFC_BLAS_TEST_ZGEMV_RM_HPP
