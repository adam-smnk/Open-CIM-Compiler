#ifndef _MLIR_INTERFACE__MEMREF__MEMREF_HPP_
#define _MLIR_INTERFACE__MEMREF__MEMREF_HPP_

#include <array>
#include <cstddef>
#include <cstdint>

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
  MemRef(MemRefDescriptor<T, rank> &memRefDesc_);

  MemRef(T *buffer, std::array<int64_t, rank> sizes);

  MemRef(T *buffer, std::array<int64_t, rank> sizes, int64_t offset,
         std::array<int64_t, rank> strides);

  virtual ~MemRef() = default;

  MemRef() = delete;

  MemRefDescriptor<T, rank> memRefDesc;
};

} // namespace memref

#include "memref.tpp"

#endif /* _MLIR_INTERFACE__MEMREF__MEMREF_HPP_ */
