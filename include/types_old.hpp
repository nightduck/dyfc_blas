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

#ifndef DYFC_BLAS_TYPES_OLD_HPP
#define DYFC_BLAS_TYPES_OLD_HPP

#include <ap_int.h>
#include <hls_stream.h>
#include <hls_vector.h>

#define MAX_BITWIDTH 4096

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

// /**
//  * A wrapper for a stream of data representing a vector.
//  *
//  * @tparam T The type of the elements in the vector. Supports any type with defined arithmetic ops.
//  * @tparam Par Number of elements retrieved in one read operation. Must be a power of 2.
//  */
// template <typename T, const unsigned int Par = MAX_BITWIDTH / 8 / sizeof(T)>
// class Vector {
//   hls::stream<WideType<T, Par>> stream_;
//   const size_t length_;

//  public:
//   // Constructors
//   /**
//    * Creates a vector with a given length. Performs checks to validate the length of parallelism
//    *
//    * @param Length The length of the vector
//    */
//   Vector(const unsigned int Length) : length_(Length) {
// #pragma HLS INLINE
//     static_assert(Par % 1 << log2(Par) == 0, "Par must be a power of 2");
// #ifndef __SYNTHESIS__
//     assert(("Cols must be greater than 0", Length > 0));
//     assert(("Length must be a multiple of Par", Length % Par == 0));
// #endif
//   }

//   /**
//    * Creates a vector with a given stream. Performs checks to validate the length of parallelism
//    *
//    * @param stream The stream to use as the underlying data structure
//    * @param Length The length of the vector
//    */
//   Vector(hls::stream<WideType<T, Par>> &stream, const unsigned int Length)
//       : stream_(stream), length_(Length) {
// #pragma HLS INLINE
//     static_assert(Par % 1 << log2(Par) == 0, "Par must be a power of 2");
// #ifndef __SYNTHESIS__
//     assert(("Cols must be greater than 0", Length > 0));
//     assert(("Length must be a multiple of Par", Length % Par == 0));
// #endif
//   }

//   /**
//    * Creates a vector and fills it with a single value
//    *
//    * @param p_Val The value to fill the vector with.
//    * @param Length The length of the vector
//    */
//   Vector(T p_Val, const unsigned int Length) : stream_(), length_(Length) {
// #pragma HLS INLINE
//     static_assert(Par % 1 << log2(Par) == 0, "Par must be a power of 2");
// #ifndef __SYNTHESIS__
//     assert(("Cols must be greater than 0", Length > 0));
//     assert(("Length must be a multiple of Par", Length % Par == 0));
// #endif
//     for (size_t i = 0; i < Length; i += Par) {
// #pragma HLS PIPELINE
//       WideType<T, Par> value;
//       for (size_t j = 0; j < Par; j++) {
// #pragma HLS UNROLL
//         value[j] = p_Val;
//       }
//       stream_.write(value);
//     }
//   }

//   /**
//    * Creates a vector and fills it with an array
//    *
//    * @param in_array The array to fill the vector with.
//    * @param length The length of the vector
//    */
//   Vector(T *in_array, const unsigned int length) : stream_(), length_(length) {
// #pragma HLS INLINE
// #pragma HLS ARRAY_PARTITION variable = in_array type = cyclic factor = Par
//     static_assert(Par % 1 << log2(Par) == 0, "Par must be a power of 2");
// #ifndef __SYNTHESIS__
//     assert(("length must be greater than 0", length > 0));
//     assert(("length must be a multiple of Par", length % Par == 0));
// #endif
//     for (size_t i = 0; i < length; i += Par) {
// #pragma HLS PIPELINE
//       WideType<T, Par> value;
//       for (size_t j = 0; j < Par; j++) {
// #pragma HLS UNROLL
//         value[j] = in_array[i + j];
//       }
//       stream_.write(value);
//     }
//   }

//   /**
//    * Reads from the underlying stream
//    *
//    * @return A WideType containing the next Par elements from the stream.
//    */
//   WideType<T, Par> read() {
// #pragma HLS INLINE
//     return stream_.read();
//   }

//   /**
//    * Writes to the underlying stream
//    *
//    * @param value The value to write to the stream.
//    */
//   void write(WideType<T, Par> value) {
// #pragma HLS INLINE
//     stream_.write(value);
//   }

//   // TODO: Rename this to write_to_memory or something
//   /**
//    * Writes the stream to memory
//    *
//    * @param value The pointer to memory to write the stream to.
//    */
//   void write(T *out_array) {
// #pragma HLS INLINE
// #pragma HLS ARRAY_PARTITION variable = out_array type = cyclic factor = Par
//     for (size_t i = 0; i < length_; i += Par) {
// #pragma HLS PIPELINE
//       WideType<T, Par> value = stream_.read();
//       for (size_t j = 0; j < Par; j++) {
// #pragma HLS UNROLL
//         out_array[i + j] = value[j];
//       }
//     }
//   }

//   /**
//    * Checks if the underlying stream is empty
//    *
//    * @return True if the stream is empty, false otherwise.
//    */
//   bool empty() {
// #pragma HLS INLINE
//     return stream_.empty();
//   }

//   /**
//    * Returns the number of elements in the underlying stream
//    *
//    * @return The number of elements in the stream. Should be less than or equal
//    * to Length
//    */
//   unsigned int size() {
// #pragma HLS INLINE
//     return stream_.size();
//   }

//   //   /**
//   //    * Splits the vector into N vectors. Consumes the contents of the vector.
//   //    *
//   //    * @param v The array of vectors to split the data into
//   //    */
//   //   template<unsigned int N>
//   //   void split(Vector<T, Par> &v[N]) {
//   // #pragma HLS INLINE
//   //     for(int i = 0; i < length_; i++) {
//   // #pragma HLS PIPELINE
//   //       WideType<T, Par> value = read();
//   //       for(int j = 0; j < N; j++) {
//   // #pragma HLS UNROLL
//   //         v[j].write(value);
//   //       }
//   //     }
//   //   }

//   /**
//    * Duplicates the vector N times. Consumes the contents of the vector.
//    *
//    * @param v The vector to duplicate the data into
//    * @param n The number of times to duplicate the vector
//    */
//   void duplicate(Vector<T, Par> &v, unsigned int n) {
// #pragma HLS INLINE
//     hls::stream<WideType<T, Par>> ring_buffer;
//     for (unsigned int i = 0; i < n; i++) {
//       for (unsigned int j = 0; j < length_; j++) {
// #pragma HLS PIPELINE
//         WideType<T, Par> value;
//         if (i == 0) {
//           value = stream_.read();
//         } else {
//           value = ring_buffer.read();
//         }
//         if (i < n - 1) {
//           ring_buffer.write(value);
//         }
//         v.write(value);
//       }
//     }
//   }

//   /**
//    * Returns the size of the vector, used for dimension checks during behavioral C synthesis
//    *
//    * @return The size of the vector
//    */
//   unsigned int shape() { return length_; }

//   // TODO: Add support for reshaping and slicing. With the former returning a
//   // matrix potentially
// };

// /**
//  * A wrapper for a stream of data representing a general matrix.
//  *
//  * @tparam T The type of the elements in the matrix. Supports any type with
//  * defined arithmetic ops.
//  * @tparam Order The major order of the matrix. Can be either RowMajor or
//  * ColMajor.
//  * @tparam Par Number of elements retrieved in one read operation. Must be a
//  * power of 2.
//  */
// template <typename T, MajorOrder Order = RowMajor,
//           const unsigned int Par = MAX_BITWIDTH / 8 / sizeof(T)>
// class Matrix {
//   hls::stream<WideType<T, Par>> stream_;
//   const unsigned int rows_;
//   const unsigned int cols_;

//  public:
//   // Constructors
//   Matrix(const unsigned int Rows, const unsigned int Cols) : rows_(Rows), cols_(Cols) {
// #pragma HLS INLINE
//     static_assert(Par % 1 << log2(Par) == 0, "Par must be a power of 2");
// #ifndef __SYNTHESIS__
//     assert(("Rows must be greater than 0", Rows > 0));
//     assert(("Cols must be greater than 0", Cols > 0));

//     // TODO: Maybe permit non multiples of Par? And just pad it with zeros
//     // internally.
//     if (Order == RowMajor) {
//       assert(("Cols must be a multiple of Par", Cols % Par == 0));
//     } else {
//       assert(("Rows must be a multiple of Par", Rows % Par == 0));
//     }
// #endif
//   }

//   Matrix(hls::stream<WideType<T, Par>> &stream, const unsigned int Rows, const unsigned int Cols)
//       : stream_(stream), rows_(Rows), cols_(Cols) {
// #pragma HLS INLINE
//     static_assert(Par % 1 << log2(Par) == 0, "Par must be a power of 2");
// #ifndef __SYNTHESIS__
//     assert(("Rows must be greater than 0", Rows > 0));
//     assert(("Cols must be greater than 0", Cols > 0));

//     // TODO: Maybe permit non multiples of Par? And just pad it with zeros
//     // internally.
//     if (Order == RowMajor) {
//       assert(("Cols must be a multiple of Par", Cols % Par == 0));
//     } else {
//       assert(("Rows must be a multiple of Par", Rows % Par == 0));
//     }
// #endif
//   }

//   /**
//    * Creates a matrix and fills it with a 2D array
//    *
//    * @param in_array The array to fill the vector with. Memory is assumed to be arranaged in order
//    * matching Order specified in the class template
//    * @param Rows The number of rows in the matrix
//    * @param Cols The number of columns in the matrix
//    */
//   Matrix(T *in_array, const unsigned int Rows, const unsigned int Cols) : rows_(Rows), cols_(Cols) {
// #pragma HLS INLINE
// #pragma HLS ARRAY_PARTITION variable = in_array type = cyclic factor = Par
//     static_assert(Par % 1 << log2(Par) == 0, "Par must be a power of 2");
// #ifndef __SYNTHESIS__
//     assert(("Rows must be greater than 0", Rows > 0));
//     assert(("Cols must be greater than 0", Cols > 0));

//     // TODO: Maybe permit non multiples of Par? And just pad it with zeros
//     // internally.
//     if (Order == RowMajor) {
//       assert(("Cols must be a multiple of Par", Cols % Par == 0));
//     } else {
//       assert(("Rows must be a multiple of Par", Rows % Par == 0));
//     }
// #endif

//     if (Order == RowMajor) {
//       for (size_t i = 0; i < Rows; i++) {
//         for (size_t j = 0; j < Cols; j += Par) {
// #pragma HLS PIPELINE
// #pragma HLS LOOP_FLATTEN
//           WideType<T, Par> value;
//           for (size_t k = 0; k < Par; k++) {
// #pragma HLS UNROLL
//             value[k] = in_array[i * Cols + j + k];
//           }
//           stream_.write(value);
//         }
//       }
//     } else {
//       for (size_t i = 0; i < Cols; i++) {
//         for (size_t j = 0; j < Rows; j += Par) {
// #pragma HLS PIPELINE
// #pragma HLS LOOP_FLATTEN
//           WideType<T, Par> value;
//           for (size_t k = 0; k < Par; k++) {
// #pragma HLS UNROLL
//             value[k] = in_array[i * Rows + j + k];
//           }
//           stream_.write(value);
//         }
//       }
//     }
//   }

//   // Matrix can't be copied.
//   Matrix(const Matrix &other) = delete;
//   Matrix &operator=(const Matrix &other) = delete;

//   // Matrix can be moved, but this consumes it
//   Matrix(Matrix &&other) noexcept
//       : stream_(std::move(other.stream_)), rows_(other.rows_), cols_(other.cols_) {
//     // TODO: Set consume flag when that's implemented
//   }
//   Matrix &operator=(Matrix &&other) noexcept {
//     if (this != &other) {
//       stream_ = other.stream_;
//       rows_ = other.rows_;
//       cols_ = other.cols_;
//     }
//     return *this;
//   }

//   /**
//    * Reads from the underlying stream
//    *
//    * @return A WideType containing the next Par elements from the stream.
//    */
//   WideType<T, Par> read() {
// #pragma HLS INLINE
//     return stream_.read();
//   }

//   /**
//    * Writes to the underlying stream
//    *
//    * @param value The value to write to the stream.
//    */
//   void write(WideType<T, Par> value) {
// #pragma HLS INLINE
//     stream_.write(value);
//   }

//   // TODO: Allow a col-major matrix to write to row-major memory and vice versa once you can figure
//   //       out why it creates an II violation (or how to appropriately pragma it)
//   // template<MajorOrder OutputOrder = Order>
//   //  * @tparam OutputOrder The major order of the memory. Can be either RowMajor or ColMajor.
//   //  *         Defaults to whatever order the matrix is in.

//   /**
//    * Writes the stream to memory.
//    *
//    *
//    * @param value The pointer to memory to write the stream to.
//    */
//   void to_memory(T *out_array) {
// #pragma HLS INLINE
// #pragma HLS ARRAY_PARTITION variable = out_array type = cyclic factor = Par
//     if (Order == RowMajor) {
//       for (size_t i = 0; i < rows_; i++) {
//         for (size_t j = 0; j < cols_; j += Par) {
// #pragma HLS PIPELINE
// #pragma HLS LOOP_FLATTEN
//           WideType<T, Par> value = stream_.read();
//           for (size_t k = 0; k < Par; k++) {
// #pragma HLS UNROLL
//             // if (OutputOrder == RowMajor) {
//             out_array[i * cols_ + j + k] = value[k];
//             // } else {
//             //   out_array[(j + k) * rows_ + i] = value[k];
//             // }
//           }
//         }
//       }
//     } else {
//       for (size_t i = 0; i < cols_; i++) {
//         for (size_t j = 0; j < rows_; j += Par) {
// #pragma HLS PIPELINE
// #pragma HLS LOOP_FLATTEN
//           WideType<T, Par> value = stream_.read();
//           for (size_t k = 0; k < Par; k++) {
// #pragma HLS UNROLL
//             // if (OutputOrder == RowMajor) {
//             //   out_array[(j + k) * cols_ + i] = value[k];
//             // } else {
//             out_array[i * rows_ + j + k] = value[k];
//             // }
//           }
//         }
//       }
//     }
//   }

//   /**
//    * Duplicates the matrix N times. Consumes the contents of the matrix.
//    *
//    * @param v The matrix to duplicate the data into
//    * @param n The number of times to duplicate the matrix
//    */
//   void duplicate(Matrix<T, Order, Par> &m, unsigned int n) {
// #pragma HLS INLINE
//     hls::stream<WideType<T, Par>> ring_buffer;
//     for (unsigned int i = 0; i < n; i++) {
//       for (unsigned int j = 0; j < this->cols_ * this->rows_; j++) {
// #pragma HLS PIPELINE
//         WideType<T, Par> value;
//         if (i == 0) {
//           value = stream_.read();
//         } else {
//           value = ring_buffer.read();
//         }
//         if (i < n - 1) {
//           ring_buffer.write(value);
//         }
//         m.write(value);
//       }
//     }
//   }

//   /**
//    * Checks if the underlying stream is empty
//    *
//    * @return True if the stream is empty, false otherwise.
//    */
//   bool empty() {
// #pragma HLS INLINE
//     return stream_.empty();
//   }

//   /**
//    * Returns the number of elements in the underlying stream
//    *
//    * @return The number of elements in the stream. Should be less than or equal
//    * to Rows * Cols
//    */
//   unsigned int size() {
// #pragma HLS INLINE
//     return stream_.size();
//   }

//   /**
//    * Returns the shape of the matrix, used for dimension checks during behavioral C synthesis
//    *
//    * @return The shape of the matrix
//    */
//   std::pair<unsigned int, unsigned int> shape() {
// #pragma HLS INLINE
//     return {rows_, cols_};
//   }

//   /**
//    * Returns the number of rows in the matrix, used for dimension checks during behavioral C
//    *
//    * @return The number of rows in the matrix
//    */
//   unsigned int rows() {
// #pragma HLS INLINE
//     return rows_;
//   }

//   /**
//    * Returns the number of columns in the matrix, used for dimension checks during behavioral C
//    *
//    * @return The number of columns in the matrix
//    */
//   unsigned int cols() {
// #pragma HLS INLINE
//     return cols_;
//   }

//   // TODO: Add support for reshaping and slicing. With the former returning a
//   // vector potentially
// };

/**
 * A wrapper for a stream of data representing a banded matrix.
 *
 * @tparam T The type of the elements in the matrix. Supports any type with
 * defined arithmetic ops.
 * @tparam SubDiagonals The number of subdiagonals in the matrix.
 * @tparam SupDiagonals The number of superdiagonals in the matrix.
 * @tparam Par Number of elements retrieved in one read operation. Must be a
 * power of 2.
 */
template <typename T, const unsigned int SubDiagonals, const unsigned int SupDiagonals,
          const unsigned int Par = MAX_BITWIDTH / 8 / sizeof(T)>
class BandedMatrix : public Matrix<T, RowMajor, Par> {};

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
template <typename T, unsigned const int Par = MAX_BITWIDTH / 8 / sizeof(T)>
using DiagonalMatrix = BandedMatrix<T, 0, 0, Par>;

/**
 * A wrapper for a stream of data representing a triangular matrix.
 * NOTE: Store packed by default?
 *
 * @tparam T The type of the elements in the matrix. Supports any type with
 * defined arithmetic ops.
 * @tparam Order The major order of the matrix. Can be either RowMajor or
 * ColMajor.
 * @tparam UpLo Whether the matrix is upper or lower triangular.
 * @tparam Par Number of elements retrieved in one read operation. Must be a
 * power of 2.
 */
template <typename T, const MajorOrder Order = RowMajor, const UpperLower UpLo = Upper,
          const unsigned int Par = MAX_BITWIDTH / 8 / sizeof(T)>
class TriangularMatrix : public Matrix<T, Order, Par> {};

/**
 * A wrapper for a stream of data representing a triangular banded matrix.
 * NOTE: Upper triangular is always row major?
 *
 * @tparam T The type of the elements in the matrix. Supports any type with
 * defined arithmetic ops.
 * @tparam Diagonals The number of subdiagonals in the matrix.
 * @tparam UpLo Whether the matrix is upper or lower triangular.
 * @tparam Par Number of elements retrieved in one read operation. Must be a
 * power of 2.
 */
template <typename T, const unsigned int Diagonals,
          const unsigned int Par = MAX_BITWIDTH / 8 / sizeof(T), const UpperLower UpLo = Upper>
class TriangularBandedMatrix : public BandedMatrix<T, (UpLo == Upper) ? 0 : Diagonals,
                                                   (UpLo == Lower) ? 0 : Diagonals, Par> {};

// TODO: Add support for unit triangular matrices and their banded equivalents

/**
 * A wrapper for a stream of data representing a symmetric matrix.
 * NOTE: Store packed by default?
 *
 * @tparam T The type of the elements in the matrix. Supports any type with
 * defined arithmetic ops.
 * @tparam Order The major order of the matrix. Can be either RowMajor or
 * ColMajor.
 * @tparam UpLo Whether the matrix is upper or lower triangular.
 * @tparam Par Number of elements retrieved in one read operation. Must be a
 * power of 2.
 */
template <typename T, const MajorOrder Order = RowMajor, const UpperLower UpLo = Upper,
          const unsigned int Par = MAX_BITWIDTH / 8 / sizeof(T)>
class SymmetricMatrix : public Matrix<T, Order, Par> {};

/**
 * A wrapper for a stream of data representing a symmetric banded matrix.
 *
 * @tparam T The type of the elements in the matrix. Supports any type with
 * defined arithmetic ops.
 * @tparam Diagonals The number of superdiagonals/subdiagonals in the matrix
 * (they are equal).
 * @tparam UpLo Whether the matrix is upper or lower triangular.
 * @tparam Par Number of elements retrieved in one read operation. Must be a
 * power of 2.
 */
template <typename T, const unsigned int Diagonals, const UpperLower UpLo = Upper,
          const unsigned int Par = MAX_BITWIDTH / 8 / sizeof(T)>
class SymmetricBandedMatrix : public BandedMatrix<T, Diagonals, Diagonals, Par> {};

/**
 * A wrapper for a stream of data representing a Hermitian matrix.
 * NOTE: Store packed by default?
 * NOTE: Throw warnings when T is not complex (because that's just a symmetric
 * matrix)
 *
 * @tparam T The type of the elements in the matrix. Supports any type with
 * defined arithmetic ops.
 * @tparam Order The major order of the matrix. Can be either RowMajor or
 * ColMajor.
 * @tparam Par Number of elements retrieved in one read operation. Must be a
 * power of 2.
 */
template <typename T, const MajorOrder Order = RowMajor, const UpperLower UpLo = Upper,
          const unsigned int Par = MAX_BITWIDTH / 8 / sizeof(T)>
class HermitianMatrix : public Matrix<T, Order, Par> {};

/**
 * A wrapper for a stream of data representing a Hermitian banded matrix.
 * NOTE: Throw warnings when T is not complex (because that's just a banded
 * symmetric matrix)
 *
 * @tparam T The type of the elements in the matrix. Supports any type with
 * defined arithmetic ops.
 * @tparam Diagonals The number of superdiagonals/subdiagonals in the matrix
 * (they are equal).
 * @tparam UpLo Whether the matrix is upper or lower triangular.
 * @tparam Par Number of elements retrieved in one read operation. Must be a
 * power of 2.
 */
template <typename T, const unsigned int Diagonals, const UpperLower UpLo = Upper,
          const unsigned int Par = MAX_BITWIDTH / 8 / sizeof(T)>
class HermitianBandedMatrix : public BandedMatrix<T, Diagonals, Diagonals, Par> {};

/**
 * Transposes a column-major matrix into a row-major matrix.
 * Doesn't move any values, just changes the type. This is analogous to the TRANSPOSE flag used in
 * the BLAS standard.
 *
 * @tparam T The type of the elements in the matrix. Supports any type with defined arithmetic ops.
 * @tparam Par Number of elements retrieved in one read operation. Must be a power of 2.
 *
 * @param A The column-major matrix to read from
 * @param AT The row-major matrix to write to
 */
template <typename T, const unsigned int Par = MAX_BITWIDTH / 8 / sizeof(T)>
void transpose(Matrix<T, ColMajor, Par> &A, Matrix<T, RowMajor, Par> &AT) {
#ifndef __SYNTHESIS__
  assert(("Dimensions of A and AT must match", A.rows() == AT.cols() && A.cols() == AT.rows()));
#endif
  for (size_t i = 0; i < A.cols(); i++) {
    for (size_t j = 0; j < A.rows(); j += Par) {
      AT.write(A.read());
    }
  }
}

/**
 * Transposes a row-major matrix into a column-major matrix.
 * Doesn't move any values, just changes the type. This is analogous to the TRANSPOSE flag used in
 *
 * @tparam T The type of the elements in the matrix. Supports any type with defined arithmetic ops.
 * @tparam Par Number of elements retrieved in one read operation. Must be a power of 2.
 *
 * @param A The row-major matrix to read from
 * @param AT The column-major matrix to write to
 */
template <typename T, const unsigned int Par = MAX_BITWIDTH / 8 / sizeof(T)>
void transpose(Matrix<T, RowMajor, Par> &A, Matrix<T, ColMajor, Par> &AT) {
#ifndef __SYNTHESIS__
  assert(("Dimensions of A and AT must match", A.rows() == AT.cols() && A.cols() == AT.rows()));
#endif
  for (size_t i = 0; i < A.rows(); i++) {
    for (size_t j = 0; j < A.cols(); j += Par) {
      AT.write(A.read());
    }
  }
}

}  // namespace blas
}  // namespace dyfc

#endif  // DYFC_BLAS_TYPES_OLD_HPP
