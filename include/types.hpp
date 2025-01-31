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

#ifndef DYFC_BLAS_TYPES_HPP
#define DYFC_BLAS_TYPES_HPP

#include <ap_int.h>
#include <hls_stream.h>
#include <hls_vector.h>

namespace dyfc {
namespace blas {

enum MajorOrder { RowMajor, ColMajor };

enum UpperLower { Upper, Lower };

// Alias to avoid confusing hls vector with blas vector
template <typename T, unsigned int Par>
using WideType = hls::vector<T, Par>;

// NOTE: A design decision was made to include the vector and matrix dimensions as class members
//    instead of template parameters. This is because the functions that will accept vectors and
//    matrices as arguments would also need to specify the dimensions as template arguments, which
//    creates multiple compiled implementations of the same function. While different
//    implementations may be needed for different levels of parallelism, a block capable of
//    processing a 128x128 matrix should also be able to process a 1024x1024 matrix. for the same
//    operation. Removing the template parameters from said function allows for a single
//    implementation to be used for all argument sizes.

constexpr size_t log2(size_t n) { return ((n < 2) ? 0 : 1 + log2(n / 2)); }

/**
 * A wrapper for a stream of data representing a vector.
 *
 * @tparam T The type of the elements in the vector. Supports any type with defined arithmetic ops.
 * @tparam Par Number of elements retrieved in one read operation. Must be a power of 2.
 */
template <typename T, unsigned int Par>
class Vector {
  hls::stream<WideType<T, Par>> data();
  const int length_;

 public:
  // Constructors
  /**
   * Creates a vector with a given length. Performs checks to validate the length of parallelism
   *
   * @param Length The length of the vector
   */
  Vector(unsigned int Length) : length_(Length) {
#pragma HLS INLINE
    static_assert(Par % 1 << log2(Par) == 0, "Par must be a power of 2");
#ifndef __SYNTHESIS__
    assert(("Cols must be greater than 0", Length > 0));
    assert(("Length must be a multiple of Par", Length % Par == 0));
#endif
  }

  /**
   * Creates a vector with a given stream. Performs checks to validate the length of parallelism
   *
   * @param data The stream to use as the underlying data structure
   * @param Length The length of the vector
   */
  Vector(hls::stream<WideType<T, Par>> &data, unsigned int Length) : data(data), length_(Length) {
#pragma HLS INLINE
    static_assert(Par % 1 << log2(Par) == 0, "Par must be a power of 2");
#ifndef __SYNTHESIS__
    assert(("Cols must be greater than 0", Length > 0));
    assert(("Length must be a multiple of Par", Length % Par == 0));
#endif
  }

  /**
   * Creates a vector and fills it with a single value
   *
   * @param p_Val The value to fill the vector with.
   * @param Length The length of the vector
   */
  Vector(T p_Val, unsigned int Length) : length_(Length) {
#pragma HLS INLINE
    static_assert(Par % 1 << log2(Par) == 0, "Par must be a power of 2");
#ifndef __SYNTHESIS__
    assert(("Cols must be greater than 0", Length > 0));
    assert(("Length must be a multiple of Par", Length % Par == 0));
#endif
    for (size_t i = 0; i < Length; i += Par) {
#pragma HLS PIPELINE
      WideType<T, Par> value;
      for (size_t j = 0; j < Par; j++) {
#pragma HLS UNROLL
        value[j] = p_Val;
      }
      data.write(value);
    }
  }

  /**
   * Creates a vector and fills it with an array
   *
   * @param in_array The array to fill the vector with.
   * @param Length The length of the vector
   */
  Vector(T *in_array, unsigned int Length) : length_(Length) {
#pragma HLS INLINE
    static_assert(Par % 1 << log2(Par) == 0, "Par must be a power of 2");
#ifndef __SYNTHESIS__
    assert(("Cols must be greater than 0", Length > 0));
    assert(("Length must be a multiple of Par", Length % Par == 0));
#endif
    for (size_t i = 0; i < Length; i += Par) {
#pragma HLS PIPELINE
      WideType<T, Par> value;
      for (size_t j = 0; j < Par; j++) {
#pragma HLS UNROLL
        value[j] = in_array[i + j];
      }
      data.write(value);
    }
  }

  /**
   * Reads from the underlying stream
   *
   * @return A WideType containing the next Par elements from the stream.
   */
  WideType<T, Par> read() {
#pragma HLS INLINE
    return data.read();
  }

  /**
   * Writes to the underlying stream
   *
   * @param value The value to write to the stream.
   */
  void write(WideType<T, Par> value) {
#pragma HLS INLINE
    data.write(value);
  }

  /**
   * Writes the stream to memory
   *
   * @param value The pointer to memory to write the stream to.
   */
  void write(T *out_array) {
#pragma HLS INLINE
    for (size_t i = 0; i < length_; i += Par) {
#pragma HLS PIPELINE
      WideType<T, Par> value = data.read();
      for (size_t j = 0; j < Par; j++) {
#pragma HLS UNROLL
        out_array[i + j] = value[j];
      }
    }
  }

  /**
   * Checks if the underlying stream is empty
   *
   * @return True if the stream is empty, false otherwise.
   */
  bool empty() {
#pragma HLS INLINE
    return data.empty();
  }

  /**
   * Returns the number of elements in the underlying stream
   *
   * @return The number of elements in the stream. Should be less than or equal
   * to Length
   */
  unsigned int size() {
#pragma HLS INLINE
    return data.size();
  }

  //   /**
  //    * Splits the vector into N vectors. Consumes the contents of the vector.
  //    *
  //    * @param v The array of vectors to split the data into
  //    */
  //   template<unsigned int N>
  //   void split(Vector<T, Par> &v[N]) {
  // #pragma HLS INLINE
  //     for(int i = 0; i < length_; i++) {
  // #pragma HLS PIPELINE
  //       WideType<T, Par> value = read();
  //       for(int j = 0; j < N; j++) {
  // #pragma HLS UNROLL
  //         v[j].write(value);
  //       }
  //     }
  //   }

  /**
   * Duplicates the vector N times. Consumes the contents of the vector.
   *
   * @param v The vector to duplicate the data into
   * @param n The number of times to duplicate the vector
   */
  void duplicate(Vector<T, Par> &v, unsigned int n) {
#pragma HLS INLINE
    hls::stream<WideType<T, Par>> ring_buffer;
    for (unsigned int i = 0; i < n; i++) {
      for (unsigned int j = 0; j < length_; j++) {
#pragma HLS PIPELINE
        WideType<T, Par> value;
        if (i == 0) {
          value = read();
        } else {
          value = ring_buffer.read();
        }
        if (i < n - 1) {
          ring_buffer.write(value);
        }
        v.write(value);
      }
    }
  }

  /**
   * Returns the size of the vector, used for dimension checks during behavioral C synthesis
   *
   * @return The size of the vector
   */
  unsigned int shape() { return length_; }

  // TODO: Add support for reshaping and slicing. With the former returning a
  // matrix potentially
};

/**
 * A wrapper for a stream of data representing a general matrix.
 *
 * @tparam T The type of the elements in the matrix. Supports any type with
 * defined arithmetic ops.
 * @tparam Par Number of elements retrieved in one read operation. Must be a
 * power of 2.
 * @tparam Rows The number of rows in the matrix.
 * @tparam Cols The number of columns in the matrix.
 * @tparam Order The major order of the matrix. Can be either RowMajor or
 * ColMajor.
 */
template <typename T, unsigned int Par, MajorOrder Order = RowMajor>
class Matrix {
  hls::stream<WideType<T, Par>> data();
  const unsigned int rows_;
  const unsigned int cols_;

 public:
  // Constructors
  Matrix(unsigned int Rows, unsigned int Cols) : rows_(Rows), cols_(Cols) {
#pragma HLS INLINE
    static_assert(Par % 1 << log2(Par) == 0, "Par must be a power of 2");
#ifndef __SYNTHESIS__
    assert(("Rows must be greater than 0", Rows > 0));
    assert(("Cols must be greater than 0", Cols > 0));

    // TODO: Maybe permit non multiples of Par? And just pad it with zeros
    // internally.
    if (Order == RowMajor) {
      assert(("Cols must be a multiple of Par", Cols % Par == 0));
    } else {
      assert(("Rows must be a multiple of Par", Rows % Par == 0));
    }
#endif
  }

  Matrix(hls::stream<WideType<T, Par>> &data, unsigned int Rows, unsigned int Cols)
      : data(data), rows_(Rows), cols_(Cols) {
#pragma HLS INLINE
    static_assert(Par % 1 << log2(Par) == 0, "Par must be a power of 2");
#ifndef __SYNTHESIS__
    assert(("Rows must be greater than 0", Rows > 0));
    assert(("Cols must be greater than 0", Cols > 0));

    // TODO: Maybe permit non multiples of Par? And just pad it with zeros
    // internally.
    if (Order == RowMajor) {
      assert(("Cols must be a multiple of Par", Cols % Par == 0));
    } else {
      assert(("Rows must be a multiple of Par", Rows % Par == 0));
    }
#endif
  }

  /**
   * Creates a matrix and fills it with a 2D array
   *
   * @param in_array The array to fill the vector with.
   * @param Length The length of the vector
   */
  Matrix(T *in_array, unsigned int Rows, unsigned int Cols) : rows_(Rows), cols_(Cols) {
#pragma HLS INLINE
    static_assert(Par % 1 << log2(Par) == 0, "Par must be a power of 2");
#ifndef __SYNTHESIS__
    assert(("Rows must be greater than 0", Rows > 0));
    assert(("Cols must be greater than 0", Cols > 0));

    // TODO: Maybe permit non multiples of Par? And just pad it with zeros
    // internally.
    if (Order == RowMajor) {
      assert(("Cols must be a multiple of Par", Cols % Par == 0));
    } else {
      assert(("Rows must be a multiple of Par", Rows % Par == 0));
    }
#endif

    if (Order == RowMajor) {
      for (size_t i = 0; i < Rows; i++) {
        for (size_t j = 0; j < Cols; j += Par) {
#pragma HLS PIPELINE
#pragma HLS LOOP_FLATTEN
          WideType<T, Par> value;
          for (size_t k = 0; k < Par; k++) {
            value[k] = in_array[i * Cols + j + k];
          }
          data.write(value);
        }
      }
    } else {
      for (size_t i = 0; i < Cols; i++) {
        for (size_t j = 0; j < Rows; j += Par) {
#pragma HLS PIPELINE
#pragma HLS LOOP_FLATTEN
          WideType<T, Par> value;
          for (size_t k = 0; k < Par; k++) {
            value[k] = in_array[i * Rows + j + k];
          }
          data.write(value);
        }
      }
    }
  }

  /**
   * Reads from the underlying stream
   *
   * @return A WideType containing the next Par elements from the stream.
   */
  WideType<T, Par> read() {
#pragma HLS INLINE
    return data.read();
  }

  /**
   * Writes to the underlying stream
   *
   * @param value The value to write to the stream.
   */
  void write(WideType<T, Par> value) {
#pragma HLS INLINE
    data.write(value);
  }

  /**
   * Checks if the underlying stream is empty
   *
   * @return True if the stream is empty, false otherwise.
   */
  bool empty() {
#pragma HLS INLINE
    return data.empty();
  }

  /**
   * Returns the number of elements in the underlying stream
   *
   * @return The number of elements in the stream. Should be less than or equal
   * to Rows * Cols
   */
  unsigned int size() {
#pragma HLS INLINE
    return data.size();
  }

  /**
   * Returns the shape of the matrix, used for dimension checks during behavioral C synthesis
   *
   * @return The shape of the matrix
   */
  std::pair<unsigned int, unsigned int> shape() {
#pragma HLS INLINE
    return {rows_, cols_};
  }

  /**
   * Returns the number of rows in the matrix, used for dimension checks during behavioral C
   *
   * @return The number of rows in the matrix
   */
  unsigned int rows() {
#pragma HLS INLINE
    return rows_;
  }

  /**
   * Returns the number of columns in the matrix, used for dimension checks during behavioral C
   *
   * @return The number of columns in the matrix
   */
  unsigned int cols() {
#pragma HLS INLINE
    return cols_;
  }

  // TODO: Add support for reshaping and slicing. With the former returning a
  // vector potentially
};

/**
 * A wrapper for a stream of data representing a banded matrix.
 *
 * @tparam T The type of the elements in the matrix. Supports any type with
 * defined arithmetic ops.
 * @tparam Par Number of elements retrieved in one read operation. Must be a
 * power of 2.
 * @tparam SubDiagonals The number of subdiagonals in the matrix.
 * @tparam SupDiagonals The number of superdiagonals in the matrix.
 */
template <typename T, unsigned int Par, unsigned int SubDiagonals, unsigned int SupDiagonals>
class BandedMatrix : public Matrix<T, Par> {};

/**
 * A wrapper for a stream of data representing a diagonal matrix. This is
 * functionally equivalent to a banded matrix with 0 subdiagonals and 0
 * superdiagonals.
 *
 * @tparam T The type of the elements in the matrix. Supports any type with
 * defined arithmetic ops.
 * @tparam Par Number of elements retrieved in one read operation. Must be a
 * power of 2.
 */
template <typename T, unsigned int Par>
using DiagonalMatrix = BandedMatrix<T, Par, 0, 0>;

/**
 * A wrapper for a stream of data representing a triangular matrix.
 * NOTE: Store packed by default?
 *
 * @tparam T The type of the elements in the matrix. Supports any type with
 * defined arithmetic ops.
 * @tparam Par Number of elements retrieved in one read operation. Must be a
 * power of 2.
 * @tparam Order The major order of the matrix. Can be either RowMajor or
 * ColMajor.
 * @tparam UpLo Whether the matrix is upper or lower triangular.
 */
template <typename T, unsigned int Par, MajorOrder Order = RowMajor, UpperLower UpLo = Upper>
class TriangularMatrix : public Matrix<T, Par, Order> {};

/**
 * A wrapper for a stream of data representing a triangular banded matrix.
 * NOTE: Upper triangular is always row major?
 *
 * @tparam T The type of the elements in the matrix. Supports any type with
 * defined arithmetic ops.
 * @tparam Par Number of elements retrieved in one read operation. Must be a
 * power of 2.
 * @tparam Diagonals The number of subdiagonals in the matrix.
 * @tparam UpLo Whether the matrix is upper or lower triangular.
 */
template <typename T, unsigned int Par, unsigned int Diagonals, UpperLower UpLo = Upper>
class TriangularBandedMatrix : public BandedMatrix<T, Par, (UpLo == Upper) ? 0 : Diagonals,
                                                   (UpLo == Lower) ? 0 : Diagonals> {};

// TODO: Add support for unit triangular matrices and their banded equivalents

/**
 * A wrapper for a stream of data representing a symmetric matrix.
 * NOTE: Store packed by default?
 *
 * @tparam T The type of the elements in the matrix. Supports any type with
 * defined arithmetic ops.
 * @tparam Par Number of elements retrieved in one read operation. Must be a
 * power of 2.
 * @tparam Order The major order of the matrix. Can be either RowMajor or
 * ColMajor.
 * @tparam UpLo Whether the matrix is upper or lower triangular.
 */
template <typename T, unsigned int Par, MajorOrder Order = RowMajor, UpperLower UpLo = Upper>
class SymmetricMatrix : public Matrix<T, Par, Order> {};

/**
 * A wrapper for a stream of data representing a symmetric banded matrix.
 *
 * @tparam T The type of the elements in the matrix. Supports any type with
 * defined arithmetic ops.
 * @tparam Par Number of elements retrieved in one read operation. Must be a
 * power of 2.
 * @tparam Diagonals The number of superdiagonals/subdiagonals in the matrix
 * (they are equal).
 * @tparam UpLo Whether the matrix is upper or lower triangular.
 */
template <typename T, unsigned int Par, unsigned int Diagonals, UpperLower UpLo = Upper>
class SymmetricBandedMatrix : public BandedMatrix<T, Par, Diagonals, Diagonals> {};

/**
 * A wrapper for a stream of data representing a Hermitian matrix.
 * NOTE: Store packed by default?
 * NOTE: Throw warnings when T is not complex (because that's just a symmetric
 * matrix)
 *
 * @tparam T The type of the elements in the matrix. Supports any type with
 * defined arithmetic ops.
 * @tparam Par Number of elements retrieved in one read operation. Must be a
 * power of 2.
 * @tparam Order The major order of the matrix. Can be either RowMajor or
 * ColMajor.
 */
template <typename T, unsigned int Par, MajorOrder Order = RowMajor, UpperLower UpLo = Upper>
class HermitianMatrix : public Matrix<T, Par, Order> {};

/**
 * A wrapper for a stream of data representing a Hermitian banded matrix.
 * NOTE: Throw warnings when T is not complex (because that's just a banded
 * symmetric matrix)
 *
 * @tparam T The type of the elements in the matrix. Supports any type with
 * defined arithmetic ops.
 * @tparam Par Number of elements retrieved in one read operation. Must be a
 * power of 2.
 * @tparam Diagonals The number of superdiagonals/subdiagonals in the matrix
 * (they are equal).
 * @tparam UpLo Whether the matrix is upper or lower triangular.
 */
template <typename T, unsigned int Par, unsigned int Diagonals, UpperLower UpLo = Upper>
class HermitianBandedMatrix : public BandedMatrix<T, Par, Diagonals, Diagonals> {};

}  // namespace blas
}  // namespace dyfc

#endif  // DYFC_BLAS_TYPES_HPP
