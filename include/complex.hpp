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

#ifndef DYFC_BLAS_COMPLEX_HPP
#define DYFC_BLAS_COMPLEX_HPP

namespace dyfc {
namespace blas {

template <typename T>
class Complex {
  T real;
  T imag;

 public:
  Complex(T real = 0, T imag = 0) : real(real), imag(imag) {}

  Complex operator+(const Complex& other) const {
    return Complex(real + other.real, imag + other.imag);
  }

  Complex operator-(const Complex& other) const {
    return Complex(real - other.real, imag - other.imag);
  }

  Complex operator*(const Complex& other) const {
    T newReal = real * other.real - imag * other.imag;
    T newImag = real * other.imag + imag * other.real;
    return Complex(newReal, newImag);
  }

  Complex operator/(const Complex& other) const {
    T denominator = other.real * other.real + other.imag * other.imag;
    T newReal = (real * other.real + imag * other.imag) / denominator;
    T newImag = (imag * other.real - real * other.imag) / denominator;
    return Complex(newReal, newImag);
  }

  // TODO: Implement assign equal operations

  Complex conj() const { return Complex(real, -imag); }
};

}  // namespace blas
}  // namespace dyfc

#endif  // DYFC_BLAS_COMPLEX_HPP
