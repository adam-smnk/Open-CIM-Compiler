#ifndef _MLIR_INTERFACE__MEMREF__MEMREF_TPP_
#define _MLIR_INTERFACE__MEMREF__MEMREF_TPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>

namespace memref {

template <typename T, size_t rank>
MemRef<T, rank>::MemRef(MemRefDescriptor<T, rank> &memRefDesc_)
    : memRefDesc(memRefDesc_) {}

template <typename T, size_t rank>
MemRef<T, rank>::MemRef(T *buffer, std::array<int64_t, rank> sizes)
    : MemRef(buffer, sizes, 0, std::array<int64_t, rank>()) {}

template <typename T, size_t rank>
MemRef<T, rank>::MemRef(T *buffer, std::array<int64_t, rank> sizes,
                        int64_t offset, std::array<int64_t, rank> strides) {
  memRefDesc.allocated = buffer;
  memRefDesc.aligned = memRefDesc.allocated;
  memRefDesc.offset = offset;
  std::memcpy(memRefDesc.sizes, sizes.data(), sizes.size() * sizeof(int64_t));
  std::memcpy(memRefDesc.strides, strides.data(),
              strides.size() * sizeof(int64_t));
}

} // namespace memref

#endif /* _MLIR_INTERFACE__MEMREF__MEMREF_TPP_ */
