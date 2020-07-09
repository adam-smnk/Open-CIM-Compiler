#include "mlir_interface/cim/cim_ops.hpp"

#include <stdlib.h>

template <typename elementType>
static void
cim_configure_crossbar_helper(int32_t tile_id,
                              memref::MemRefDescriptor<elementType, 2> *B) {
  // C[M][N] =  A[M][K] * B[K][N]
  const uint32_t N = B->sizes[1];
  const uint32_t K = B->sizes[0];

  // TODO: Add calls to CIM library to get biases and proper B mapping
  uint8_t *bias = (uint8_t *)malloc(N * sizeof(elementType));

  cim_configure_crossbar(tile_id, (uint8_t **)&B->allocated[B->offset],
                         (uint8_t *)bias, N, K);

  free(bias);
}

template <typename elementType>
static void cim_gemm_helper(int32_t tile_id,
                            memref::MemRefDescriptor<elementType, 2> *A,
                            memref::MemRefDescriptor<elementType, 2> *C) {
  // C[M][N] =  A[M][K] * B[K][N]
  const uint32_t M = C->sizes[0];
  const uint32_t N = C->sizes[1];
  const uint32_t K = A->sizes[1];

  cim_gemm(tile_id, (uint8_t *)&A->allocated[A->offset], M, N, K, K, 0, 0,
           (uint8_t *)&C->allocated[C->offset], N);
}

template <typename elementType>
static void cim_gevm_helper(int32_t tile_id,
                            memref::MemRefDescriptor<elementType, 1> *A,
                            memref::MemRefDescriptor<elementType, 1> *C) {
  // C[N] =  A[K] * B[K][N]
  const uint32_t N = C->sizes[0];
  const uint32_t K = A->sizes[0];

  cim_gevm(tile_id, (uint8_t *)&A->allocated[A->offset], K, N, 0, 0,
           (uint8_t *)&C->allocated[C->offset]);
}

// MLIR cim.barrier
void _mlir_ciface_cim_barrier(int32_t tile_id) { cim_await(tile_id); }

// MLIR cim.write_to_crossbar
void _mlir_ciface_cim_write_to_crossbar_i8(
    int32_t tile_id, memref::MemRefDescriptor<int8_t, 2> *B) {
  cim_configure_crossbar_helper<int8_t>(tile_id, B);
}

void _mlir_ciface_cim_write_to_crossbar_i16(
    int32_t tile_id, memref::MemRefDescriptor<int16_t, 2> *B) {
  cim_configure_crossbar_helper<int16_t>(tile_id, B);
}

void _mlir_ciface_cim_write_to_crossbar_i32(
    int32_t tile_id, memref::MemRefDescriptor<int32_t, 2> *B) {
  cim_configure_crossbar_helper<int32_t>(tile_id, B);
}

// MLIR cim.gemm
void _mlir_ciface_cim_gemm_i8(int32_t tile_id,
                              memref::MemRefDescriptor<int8_t, 2> *A,
                              memref::MemRefDescriptor<int8_t, 2> *C) {
  cim_gemm_helper<int8_t>(tile_id, A, C);
}

void _mlir_ciface_cim_gemm_i16(int32_t tile_id,
                               memref::MemRefDescriptor<int16_t, 2> *A,
                               memref::MemRefDescriptor<int16_t, 2> *C) {
  cim_gemm_helper<int16_t>(tile_id, A, C);
}

void _mlir_ciface_cim_gemm_i32(int32_t tile_id,
                               memref::MemRefDescriptor<int32_t, 2> *A,
                               memref::MemRefDescriptor<int32_t, 2> *C) {
  cim_gemm_helper<int32_t>(tile_id, A, C);
}

// MLIR cim.gevm
void _mlir_ciface_cim_gevm_i8(int32_t tile_id,
                              memref::MemRefDescriptor<int8_t, 1> *A,
                              memref::MemRefDescriptor<int8_t, 1> *C) {
  cim_gevm_helper<int8_t>(tile_id, A, C);
}

void _mlir_ciface_cim_gevm_i16(int32_t tile_id,
                               memref::MemRefDescriptor<int16_t, 1> *A,
                               memref::MemRefDescriptor<int16_t, 1> *C) {
  cim_gevm_helper<int16_t>(tile_id, A, C);
}

void _mlir_ciface_cim_gevm_i32(int32_t tile_id,
                               memref::MemRefDescriptor<int32_t, 1> *A,
                               memref::MemRefDescriptor<int32_t, 1> *C) {
  cim_gevm_helper<int32_t>(tile_id, A, C);
}
