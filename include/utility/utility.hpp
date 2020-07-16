#ifndef _UTILITY__UTILITY_HPP_
#define _UTILITY__UTILITY_HPP_

#include "mlir_interface/memref/memref.hpp"

namespace utility {

template <typename T, size_t rank>
void printTensor(const memref::MemRef<T, rank> &memRef);

template <typename T>
void printMatrix(const memref::MemRef<T, 2> &memRef);

template <typename T>
void printMatrix(const memref::MemRefDescriptor<T, 2> &mat);

template <typename T>
void printVector(const memref::MemRef<T, 1> &memRef);

template <typename T>
void printVector(const memref::MemRefDescriptor<T, 1> &vec);

} // namespace utility

#include "utility.tpp"

#endif /* _UTILITY__UTILITY_HPP_ */
