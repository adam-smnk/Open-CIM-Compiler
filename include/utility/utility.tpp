#ifndef _UTILITY__UTILITY_TPP_
#define _UTILITY__UTILITY_TPP_

#include "utility/utility.hpp"

#include <iostream>

namespace utility {

template <typename T>
void printMatrix(const memref::MemRef<T, 2> &memRef) {
  printMatrix(memRef.memRefDesc);
}

template <typename T>
void printMatrix(const memref::MemRefDescriptor<T, 2> &mat) {
  const int numRows = mat.sizes[0];
  const int numCols = mat.sizes[1];

  for (int i = 0; i < numRows; ++i) {
    std::cout << "| ";
    for (int j = 0; j < numCols; ++j) {
      std::cout << mat.allocated[i * numCols + j] << " ";
    }
    std::cout << "|\n";
  }
}

} // namespace utility

#endif /* _UTILITY__UTILITY_TPP_ */
