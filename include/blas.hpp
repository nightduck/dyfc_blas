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

#ifndef DYFC_BLAS_HPP
#define DYFC_BLAS_HPP

#include "assert.hpp"
#include "complex.hpp"
#include "types.hpp"
// #include "prototypes.hpp"

#include "axpy.hpp"
#include "asum.hpp"
#include "dot.hpp"
#include "mm.hpp"
#include "mv.hpp"
#include "prefixsum.hpp"

// Converts a 2D array to an unstructured pointer
#define FLATTEN_MATRIX(x) &x[0][0]

#endif  // DYFC_BLAS_HPP