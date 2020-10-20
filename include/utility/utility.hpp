#ifndef _UTILITY__UTILITY_HPP_
#define _UTILITY__UTILITY_HPP_

#include "mlir_interface/memref/memref.hpp"

namespace utility {

template <typename T, size_t rank>
void printTensor(const memref::MemRef<T, rank> &memRef);

template <typename T>
void printMatrix3D(const memref::MemRef<T, 3> &memRef);

template <typename T>
void printMatrix3D(const memref::MemRefDescriptor<T, 3> &mat);

template <typename T>
void printMatrix(const memref::MemRef<T, 2> &memRef);

template <typename T>
void printMatrix(const memref::MemRefDescriptor<T, 2> &mat);

template <typename T>
void printVector(const memref::MemRef<T, 1> &memRef);

template <typename T>
void printVector(const memref::MemRefDescriptor<T, 1> &vec);

template <typename T, size_t rank>
void printDimensions(const memref::MemRef<T, rank> &memRef);

template <typename T, size_t rank>
int tensorSize(const std::array<T, rank> &sizes);

template <typename T>
void computeGemm(const memref::MemRef<T, 2> &A, const memref::MemRef<T, 2> &B,
                 const memref::MemRef<T, 2> &C);

template <typename T>
void computeGemm(const memref::MemRefDescriptor<T, 2> &A,
                 const memref::MemRefDescriptor<T, 2> &B,
                 const memref::MemRefDescriptor<T, 2> &C);

} // namespace utility

#include "utility.tpp"

#endif /* _UTILITY__UTILITY_HPP_ */
