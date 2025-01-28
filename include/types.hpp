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

#include <hls_stream.h>
#include <hls_vector.h>
#include <ap_int.h>

namespace dyfc {
namespace blas {

enum MajorOrder {
    RowMajor,
    ColMajor
};

enum UpperLower {
    Upper,
    Lower
};

// Alias to avoid confusing hls vector with blas vector
template<typename T, unsigned int Par>
using WideType<T, Par> = hls::vector<T, Par>;

constexpr size_t log2(size_t n) {
    return ((n < 2) ? 0 : 1 + log2(n / 2));
}

/**
 * A wrapper for a stream of data representing a vector.
 * 
 * @tparam T The type of the elements in the vector. Supports any type with defined arithmetic ops.
 * @tparam Par Number of elements retrieved in one read operation. Must be a power of 2.
 * @tparam Length The length of the vector.
*/
template<typename T, unsigned int Par, unsigned int Length>
class Vector {
    hls::stream<WideType<T, Par>> data;

public:
    // Constructors
    Vector() {
#pragma HLS INLINE
        static_assert(Length > 0, "Cols must be greater than 0");
        static_assert(Par % 1 << log2(Par) == 0, "Par must be a power of 2");
        static_assert(Length % Par == 0, "Cols must be a multiple of Par");
    }

    Vector(hls::stream<WideType<T, Par>> &data) : data(data) {
#pragma HLS INLINE
        static_assert(Length > 0, "Cols must be greater than 0");
        static_assert(Par % 1 << log2(Par) == 0, "Par must be a power of 2");
        static_assert(Length % Par == 0, "Cols must be a multiple of Par");
    }

    /**
     * Fills the vector with a single value
     * 
     * @param p_Val The value to fill the vector with.
     */
    Vector(T p_Val) {
#pragma HLS INLINE
        for (size_t i = 0; i < Length; i+=Par)
        {
#pragma HLS PIPELINE
            WideType<T, Par> value;
            for (size_t j = 0; j < Par; j++)
            {
#pragma HLS UNROLL
                value[j] = p_Val;
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
     * @return The number of elements in the stream. Should be less than or equal to Length
     */
    unsigned int size() {
#pragma HLS INLINE
        return data.size();
    }

    // TODO: Add support for reshaping and slicing. With the former returning a matrix potentially
}

/**
 * A wrapper for a stream of data representing a general matrix.
 * 
 * @tparam T The type of the elements in the matrix. Supports any type with defined arithmetic ops.
 * @tparam Par Number of elements retrieved in one read operation. Must be a power of 2.
 * @tparam Rows The number of rows in the matrix.
 * @tparam Cols The number of columns in the matrix.
 * @tparam Order The major order of the matrix. Can be either RowMajor or ColMajor.
*/
template<typename T, unsigned int Par, unsigned int Rows, unsigned int Cols, MajorOrder Order=RowMajor>
class Matrix {
    hls::stream<WideType<T, Par>> data;

public:
    // Constructors
    Matrix() {
#pragma HLS INLINE
        static_assert(Rows > 0, "Rows must be greater than 0");
        static_assert(Cols > 0, "Cols must be greater than 0");
        static_assert(Par % 1 << log2(Par) == 0, "Par must be a power of 2");

        // TODO: Maybe permit non multiples of Par? And just pad it with zeros internally.
        if (Order == RowMajor) {
            static_assert(Cols % Par == 0, "Cols must be a multiple of Par");
        } else {
            static_assert(Rows % Par == 0, "Rows must be a multiple of Par");
        }
    }

    Matrix(hls::stream<WideType<T, Par>> &data) : data(data) {
#pragma HLS INLINE
        static_assert(Rows > 0, "Rows must be greater than 0");
        static_assert(Cols > 0, "Cols must be greater than 0");
        static_assert(Par % 1 << log2(Par) == 0, "Par must be a power of 2");

        // TODO: Maybe permit non multiples of Par? And just pad it with zeros internally.
        if (Order == RowMajor) {
            static_assert(Cols % Par == 0, "Cols must be a multiple of Par");
        } else {
            static_assert(Rows % Par == 0, "Rows must be a multiple of Par");
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
     * @return The number of elements in the stream. Should be less than or equal to Rows * Cols
     */
    unsigned int size() {
#pragma HLS INLINE
        return data.size();
    }

    // TODO: Add support for reshaping and slicing. With the former returning a vector potentially
}

/**
 * A wrapper for a stream of data representing a banded matrix.
 * 
 * @tparam T The type of the elements in the matrix. Supports any type with defined arithmetic ops.
 * @tparam Par Number of elements retrieved in one read operation. Must be a power of 2.
 * @tparam Rows The number of rows in the matrix.
 * @tparam Cols The number of columns in the matrix.
 * @tparam SubDiagonals The number of subdiagonals in the matrix.
 * @tparam SupDiagonals The number of superdiagonals in the matrix.
*/
template<typename T, unsigned int Par, unsigned int Rows, unsigned int Cols, unsigned int SubDiagonals, unsigned int SupDiagonals>
class BandedMatrix : public Matrix<T, Par, Rows, Cols>;

/**
 * A wrapper for a stream of data representing a diagonal matrix. This is functionally equivalent
 * to a banded matrix with 0 subdiagonals and 0 superdiagonals.
 * 
 * @tparam T The type of the elements in the matrix. Supports any type with defined arithmetic ops.
 * @tparam Par Number of elements retrieved in one read operation. Must be a power of 2.
 * @tparam Rows The number of rows in the matrix.
 * @tparam Cols The number of columns in the matrix.
*/
template<typename T, unsigned int Par, unsigned int Rows, unsigned int Cols>
using DiagonalMatrix = BandedMatrix<T, Par, Rows, Cols, 0, 0>;

/**
 * A wrapper for a stream of data representing a triangular matrix.
 * NOTE: Store packed by default?
 * 
 * @tparam T The type of the elements in the matrix. Supports any type with defined arithmetic ops.
 * @tparam Par Number of elements retrieved in one read operation. Must be a power of 2.
 * @tparam Rows The number of rows (and cols) in the matrix, which is square
 * @tparam Order The major order of the matrix. Can be either RowMajor or ColMajor.
 * @tparam UpLo Whether the matrix is upper or lower triangular.
*/
template<typename T, unsigned int Par, unsigned int Rows, MajorOrder Order=RowMajor, UpperLower UpLo=Upper>
class TriangularMatrix : public Matrix<T, Par, Rows, Cols>;

/**
 * A wrapper for a stream of data representing a triangular banded matrix.
 * NOTE: Upper triangular is always row major?
 * 
 * @tparam T The type of the elements in the matrix. Supports any type with defined arithmetic ops.
 * @tparam Par Number of elements retrieved in one read operation. Must be a power of 2.
 * @tparam Rows The number of rows (and cols) in the matrix, which is square
 * @tparam Diagonals The number of subdiagonals in the matrix.
 * @tparam UpLo Whether the matrix is upper or lower triangular.
*/
template<typename T, unsigned int Par, unsigned int Rows, unsigned int Diagonals, UpperLower UpLo=Upper>
class TriangularBandedMatrix : public BandedMatrix<T, Par, Rows, Rows, constexpr (UpLo == Upper) ? 0 : Diagonals, constexpr (UpLo == Lower) ? 0 : Diagonals>;

// TODO: Add support for unit triangular matrices and their banded equivalents

/**
 * A wrapper for a stream of data representing a symmetric matrix.
 * NOTE: Store packed by default?
 * 
 * @tparam T The type of the elements in the matrix. Supports any type with defined arithmetic ops.
 * @tparam Par Number of elements retrieved in one read operation. Must be a power of 2.
 * @tparam Rows The number of rows (and cols) in the matrix, which is square
 * @tparam Order The major order of the matrix. Can be either RowMajor or ColMajor.
*/
template<typename T, unsigned int Par, unsigned int Rows, MajorOrder Order=RowMajor, UpperLower UpLo=Upper>
class SymmetricMatrix : public Matrix<T, Par, Rows, Cols>;

/**
 * A wrapper for a stream of data representing a symmetric banded matrix.
 * 
 * @tparam T The type of the elements in the matrix. Supports any type with defined arithmetic ops.
 * @tparam Par Number of elements retrieved in one read operation. Must be a power of 2.
 * @tparam Rows The number of rows (and cols) in the matrix, which is square
 * @tparam Diagonals The number of superdiagonals/subdiagonals in the matrix (they are equal).
 * @tparam UpLo Whether the matrix is upper or lower triangular.
*/
template<typename T, unsigned int Par, unsigned int Rows, unsigned int Diagonals, UpperLower UpLo=Upper>
class SymmetricBandedMatrix : public BandedMatrix<T, Par, Rows, Rows, cDiagonals, Diagonals>;

/**
 * A wrapper for a stream of data representing a Hermitian matrix.
 * NOTE: Store packed by default?
 * NOTE: Throw warnings when T is not complex (because that's just a symmetric matrix)
 * 
 * @tparam T The type of the elements in the matrix. Supports any type with defined arithmetic ops.
 * @tparam Par Number of elements retrieved in one read operation. Must be a power of 2.
 * @tparam Rows The number of rows (and cols) in the matrix, which is square
 * @tparam Order The major order of the matrix. Can be either RowMajor or ColMajor.
*/
template<typename T, unsigned int Par, unsigned int Rows, MajorOrder Order=RowMajor, UpperLower UpLo=Upper>
class HermitianMatrix : public Matrix<T, Par, Rows, Cols>;

/**
 * A wrapper for a stream of data representing a Hermitian banded matrix.
 * NOTE: Throw warnings when T is not complex (because that's just a banded symmetric matrix)
 * 
 * @tparam T The type of the elements in the matrix. Supports any type with defined arithmetic ops.
 * @tparam Par Number of elements retrieved in one read operation. Must be a power of 2.
 * @tparam Rows The number of rows (and cols) in the matrix, which is square
 * @tparam Diagonals The number of superdiagonals/subdiagonals in the matrix (they are equal).
 * @tparam UpLo Whether the matrix is upper or lower triangular.
*/
template<typename T, unsigned int Par, unsigned int Rows, unsigned int Diagonals, UpperLower UpLo=Upper>
class HermitianBandedMatrix : public BandedMatrix<T, Par, Rows, Rows, Diagonals, Diagonals>;

} // namespace blas
} // namespace dyfc

#endif // DYFC_BLAS_TYPES_HPP
