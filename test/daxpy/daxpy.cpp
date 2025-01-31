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

#include "daxpy.hpp"
#include "blas.hpp"

void daxpy(unsigned int n, double alpha, double x[128], double y[128], double r[128]) {
    dyfc::blas::Vector<double, 64> x_v(x, 128);
    dyfc::blas::Vector<double, 64> y_v(y, 128);
    dyfc::blas::Vector<double, 64> r_v(128);

    dyfc::blas::axpy(n, alpha, x_v, y_v, r_v);

    r_v.write(r);

    return;
}
