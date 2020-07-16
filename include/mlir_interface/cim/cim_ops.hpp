#ifndef _MLIR_INTERFACE__CIM__CIM_OPS_HPP_
#define _MLIR_INTERFACE__CIM__CIM_OPS_HPP_

#include "cim_sim_mock/cim.h"
#include "mlir_interface/memref/memref.hpp"

#include <cstdint>
#include <stdlib.h>

extern "C" {

// MLIR cim.barrier
void _mlir_ciface_cim_barrier(int32_t tile_id);

// MLIR cim.write_to_crossbar
void _mlir_ciface_cim_write_to_crossbar_i8(
    int32_t tile_id, memref::MemRefDescriptor<int8_t, 2> *B);

void _mlir_ciface_cim_write_to_crossbar_i16(
    int32_t tile_id, memref::MemRefDescriptor<int16_t, 2> *B);

void _mlir_ciface_cim_write_to_crossbar_i32(
    int32_t tile_id, memref::MemRefDescriptor<int32_t, 2> *B);

// MLIR cim.gemm
void _mlir_ciface_cim_gemm_i8(int32_t tile_id,
                              memref::MemRefDescriptor<int8_t, 2> *A,
                              memref::MemRefDescriptor<int8_t, 2> *C);

void _mlir_ciface_cim_gemm_i16(int32_t tile_id,
                               memref::MemRefDescriptor<int16_t, 2> *A,
                               memref::MemRefDescriptor<int16_t, 2> *C);

void _mlir_ciface_cim_gemm_i32(int32_t tile_id,
                               memref::MemRefDescriptor<int32_t, 2> *A,
                               memref::MemRefDescriptor<int32_t, 2> *C);

// MLIR cim.gevm
void _mlir_ciface_cim_gevm_i8(int32_t tile_id,
                              memref::MemRefDescriptor<int8_t, 1> *A,
                              memref::MemRefDescriptor<int8_t, 1> *C);

void _mlir_ciface_cim_gevm_i16(int32_t tile_id,
                               memref::MemRefDescriptor<int16_t, 1> *A,
                               memref::MemRefDescriptor<int16_t, 1> *C);

void _mlir_ciface_cim_gevm_i32(int32_t tile_id,
                               memref::MemRefDescriptor<int32_t, 1> *A,
                               memref::MemRefDescriptor<int32_t, 1> *C);

} /* extern C */

#endif /* _MLIR_INTERFACE__CIM__CIM_OPS_HPP_ */
