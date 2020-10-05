#include "mlir_interface/memref/memref.hpp"
#include "simulator_interface/cim_sim.h"
#include "utility/utility.hpp"

#include <array>
#include <cstdint>
#include <iostream>

#define CIM_PRECISION int8_t

#ifndef DIM_SIZE
#define DIM_SIZE 3
#endif // DIM_SIZE

#ifndef KERNEL_SIZE
#define KERNEL_SIZE 2
#endif // KERNEL_SIZE

// Functions generated from TC
extern "C" {
void _mlir_ciface_conv1d(memref::MemRefDescriptor<CIM_PRECISION, 3> *A,
                         memref::MemRefDescriptor<CIM_PRECISION, 3> *B,
                         memref::MemRefDescriptor<CIM_PRECISION, 3> *C);
}

int main() {
  simulator_init();

  // Conv1D
  // C[N][K][H] =  A[N][C][H] * B[K][C][KH]
  const size_t rank = 3;
  const int32_t N = 1;
  const int32_t K = 3;
  const int32_t C = 3;
  const int32_t H = DIM_SIZE;
  const int32_t KH = KERNEL_SIZE;

  CIM_PRECISION matA[N][C][H];
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      for (int h = 0; h < H; ++h) {
        matA[n][c][h] = h;
      }
    }
  }

  CIM_PRECISION matB[K][C][KH];
  for (int k = 0; k < K; ++k) {
    for (int c = 0; c < C; ++c) {
      for (int kh = 0; kh < KH; ++kh) {
        matB[k][c][kh] = 1;
      }
    }
  }

  CIM_PRECISION matC[N][K][H];
  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K; ++k) {
      for (int h = 0; h < H; ++h) {
        matC[n][k][h] = 0;
      }
    }
  }

  memref::MemRef<CIM_PRECISION, rank> A((CIM_PRECISION *)matA, {N, C, H});
  memref::MemRef<CIM_PRECISION, rank> B((CIM_PRECISION *)matB, {K, C, KH});
  memref::MemRef<CIM_PRECISION, rank> memC((CIM_PRECISION *)matC, {N, K, H});

  std::cout << "A input:\n";
  utility::printTensor(A);

  std::cout << "B input:\n";
  utility::printTensor(B);

  simulator_mark_start();
#ifdef BENCH_BUILD_ARM
  const int paddingRows = KH - 1;
  CIM_PRECISION matPaddedA[N][K][H + paddingRows] = {0};
  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K; ++k) {
      for (int h = 0; h < H; ++h) {
        matPaddedA[n][k][h + paddingRows / 2 + (paddingRows % 2)] =
            matA[n][k][h];
      }
    }
  }
  memref::MemRef<CIM_PRECISION, rank> paddedA((CIM_PRECISION *)matPaddedA,
                                              {N, C, H + paddingRows});
  _mlir_ciface_conv1d(&paddedA.memRefDesc, &B.memRefDesc, &memC.memRefDesc);
#else
  _mlir_ciface_conv1d(&A.memRefDesc, &B.memRefDesc, &memC.memRefDesc);
#endif // BENCH_ARM_BUILD
  simulator_mark_end();

  std::cout << "C output:\n";
  utility::printTensor(memC);

  simulator_terminate();
}
