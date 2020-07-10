#ifndef _MLIR_INTERFACE__MEMREF__MEMREF_H_
#define _MLIR_INTERFACE__MEMREF__MEMREF_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace memref {

template <typename elementType, size_t memrefRank>
struct MemRefDescriptor {
  elementType *allocated;
  elementType *aligned;
  int64_t offset;
  int64_t sizes[memrefRank];
  int64_t strides[memrefRank];
};

// Wrapper around MLIR memref type
template <typename T, size_t rank>
class MemRef {
public:
  // Performs shallow copy, takes ownership of buffer
  MemRef(MemRefDescriptor<T, rank> &memRefDesc_);

  // Automatically allocates memory
  MemRef(std::array<int32_t, rank> sizes);

  // Takes over given buffer
  MemRef(std::unique_ptr<T> buffer, std::array<int32_t, rank> sizes);

  // Takes over given buffer
  MemRef(std::unique_ptr<T> buffer, std::array<int32_t, rank> sizes,
         int32_t offset, std::array<int32_t, rank> strides);

  virtual ~MemRef();

  MemRef() = delete;

  T *releaseMemRef();

  MemRefDescriptor<T, rank> memRefDesc;
};

} // namespace memref

#include "memref.tpp"

#endif /* _MLIR_INTERFACE__MEMREF__MEMREF_H_ */