#ifndef _UTILITY__UTILITY_TPP_
#define _UTILITY__UTILITY_TPP_

#include "utility/utility.hpp"

#include <cstdio>
#include <iostream>

namespace utility {

template <typename T, size_t rank>
void printTensor(const memref::MemRef<T, rank> &memRef) {
  std::cout << "Cannot print - unsupported tensor rank\n";
  std::cout << "Tensor dims: ";
  for (auto dim : memRef.memRefDesc.sizes) {
    std::cout << dim << " ";
  }
  std::cout << "\n";
}

template <typename T>
void printTensor(const memref::MemRef<T, 1> &memRef) {
  printVector(memRef);
}

template <typename T>
void printTensor(const memref::MemRef<T, 2> &memRef) {
  printMatrix(memRef);
}

template <typename T>
void printMatrix(const memref::MemRef<T, 2> &memRef) {
  printMatrix(memRef.memRefDesc);
}

template <typename T>
void printMatrix(const memref::MemRefDescriptor<T, 2> &mat) {
  const int numRows = mat.sizes[0];
  const int numCols = mat.sizes[1];

  for (int i = 0; i < numRows; ++i) {
    printf("| ");

    for (int j = 0; j < numCols; ++j) {
      printf("%d ", mat.allocated[i * numCols + j]);
    }
    printf("|\n");
  }
}

template <typename T>
void printVector(const memref::MemRef<T, 1> &memRef) {
  printVector(memRef.memRefDesc);
}

template <typename T>
void printVector(const memref::MemRefDescriptor<T, 1> &vec) {
  printf("| ");

  for (int i = 0; i < vec.sizes[0]; ++i) {
    printf("%d ", vec.allocated[i]);
  }
  printf("|\n");
}

} // namespace utility

#endif /* _UTILITY__UTILITY_TPP_ */
