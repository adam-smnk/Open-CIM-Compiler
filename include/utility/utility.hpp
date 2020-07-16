#ifndef _UTILITY__UTILITY_HPP_
#define _UTILITY__UTILITY_HPP_

#include "mlir_interface/memref/memref.hpp"

namespace utility {

template <typename T>
void printMatrix(const memref::MemRef<T, 2> &memRef);

template <typename T>
void printMatrix(const memref::MemRefDescriptor<T, 2> &mat);

} // namespace utility

#include "utility.tpp"

#endif /* _UTILITY__UTILITY_HPP_ */
