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
MemRef<T, rank>::MemRef(std::array<int32_t, rank> sizes)
    : MemRef(nullptr, sizes) {}

template <typename T, size_t rank>
MemRef<T, rank>::MemRef(std::unique_ptr<T> buffer,
                        std::array<int32_t, rank> sizes)
    : MemRef(std::move(buffer), sizes, 0, std::array<int32_t, rank>()) {}

template <typename T, size_t rank>
MemRef<T, rank>::MemRef(std::unique_ptr<T> buffer,
                        std::array<int32_t, rank> sizes, int32_t offset,
                        std::array<int32_t, rank> strides) {
  if (buffer) {
    memRefDesc.allocated = buffer.release();
  } else {
    int32_t bufferSize = sizes[0];
    for (int i = 1; i < sizes.size(); ++i) {
      bufferSize *= sizes[i];
    }

    memRefDesc.allocated = new T[bufferSize];
  }

  memRefDesc.aligned = memRefDesc.allocated;
  memRefDesc.offset = offset;
  std::memcpy(memRefDesc.sizes, sizes.data(), sizes.size() * sizeof(T));
  std::memcpy(memRefDesc.strides, strides.data(), strides.size() * sizeof(T));
}

template <typename T, size_t rank>
MemRef<T, rank>::~MemRef() {
  delete memRefDesc.allocated;
}

template <typename T, size_t rank>
T *MemRef<T, rank>::releaseMemRef() {
  T *buffer = memRefDesc.allocated;

  memRefDesc.allocated = nullptr;
  memRefDesc.aligned = nullptr;

  return buffer;
}

} // namespace memref

#endif /* _MLIR_INTERFACE__MEMREF__MEMREF_TPP_ */