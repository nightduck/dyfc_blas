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

// This provides an implementation of complex numbers optimized for HLS.
// The standard C implementation of std::complex is not synthesizable in HLS.

#include <iostream>
namespace dyfc {
namespace blas {

template <typename T>
class Complex {
 public:
  T real;
  T imag;

  using value_type = T;

  Complex(T real = 0, T imag = 0) : real(real), imag(imag) {}

  Complex operator+(const Complex& other) const {
#pragma HLS INLINE
    return Complex(real + other.real, imag + other.imag);
  }

  Complex operator-(const Complex& other) const {
#pragma HLS INLINE
    return Complex(real - other.real, imag - other.imag);
  }

  Complex operator*(const Complex& other) const {
#pragma HLS INLINE
    T newReal = real * other.real - imag * other.imag;
    T newImag = real * other.imag + imag * other.real;
    return Complex(newReal, newImag);
  }

  Complex operator/(const Complex& other) const {
#pragma HLS INLINE
    T denominator = other.real * other.real + other.imag * other.imag;
    T newReal = (real * other.real + imag * other.imag) / denominator;
    T newImag = (imag * other.real - real * other.imag) / denominator;
    return Complex(newReal, newImag);
  }

  bool operator==(const Complex& other) const {
#pragma HLS INLINE
    return real == other.real && imag == other.imag;
  }

  bool operator!=(const Complex& other) const {
#pragma HLS INLINE
    return real != other.real || imag != other.imag;
  }

  Complex& operator=(const Complex& other) {
#pragma HLS INLINE
    real = other.real;
    imag = other.imag;
    return *this;
  }

  Complex& operator+=(const Complex& other) {
#pragma HLS INLINE
    real += other.real;
    imag += other.imag;
    return *this;
  }

  Complex& operator-=(const Complex& other) {
#pragma HLS INLINE
    real -= other.real;
    imag -= other.imag;
    return *this;
  }

  Complex& operator*=(const Complex& other) {
#pragma HLS INLINE
    T newReal = real * other.real - imag * other.imag;
    T newImag = real * other.imag + imag * other.real;
    real = newReal;
    imag = newImag;
    return *this;
  }
  Complex& operator/=(const Complex& other) {
#pragma HLS INLINE
    T denominator = other.real * other.real + other.imag * other.imag;
    T newReal = (real * other.real + imag * other.imag) / denominator;
    T newImag = (imag * other.real - real * other.imag) / denominator;
    real = newReal;
    imag = newImag;
    return *this;
  }

  T abs() const {
#pragma HLS INLINE
    return sqrt(real * real + imag * imag);
  }
  T arg() const {
#pragma HLS INLINE
    return atan2(imag, real);
  }
  T norm() const {
#pragma HLS INLINE
    return real * real + imag * imag;
  }
  Complex conj() const {
#pragma HLS INLINE
    return Complex(real, -imag);
  }
  Complex polar(T rho, T theta) {
#pragma HLS INLINE
    return Complex(rho * cos(theta), rho * sin(theta));
  }
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const Complex<T>& c) {
  os << c.real << " + " << c.imag << "i";
  return os;
}

// TODO: Copy the following functions from members into non-member equivalents:
// abs(Complex<T> z)
// arg(Complex<T> z)
// norm(Complex<T> z)
// conj(Complex<T> z)
// polar(T rho, T theta)

// TODO: Define the following functions from the C stdlib
// proj(Complex<T> z)
// exp(Complex<T> z)
// log(Complex<T> z)
// log10(Complex<T> z)
// pow(Complex<T> z, T exponent)
// sqrt(Complex<T> z)
// sin(Complex<T> z)
// cos(Complex<T> z)
// tan(Complex<T> z)
// sinh(Complex<T> z)
// cosh(Complex<T> z)
// tanh(Complex<T> z)
// asin(Complex<T> z)
// acos(Complex<T> z)
// atan(Complex<T> z)
// asinh(Complex<T> z)
// acosh(Complex<T> z)
// atanh(Complex<T> z)

}  // namespace blas
}  // namespace dyfc

#endif  // DYFC_BLAS_COMPLEX_HPP
