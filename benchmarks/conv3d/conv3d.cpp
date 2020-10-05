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
void _mlir_ciface_conv3d(memref::MemRefDescriptor<CIM_PRECISION, 5> *A,
                         memref::MemRefDescriptor<CIM_PRECISION, 5> *B,
                         memref::MemRefDescriptor<CIM_PRECISION, 5> *C);
}

int main() {
  simulator_init();

  // Conv3D
  // C[N][K][H][W][D] =  A[N][C][H][W][D] * B[K][C][KH][KW][KD]
  const size_t rank = 5;
  const int32_t N = 1;
  const int32_t K = 3;
  const int32_t C = 3;
  const int32_t H = DIM_SIZE;
  const int32_t W = DIM_SIZE;
  const int32_t D = DIM_SIZE;
  const int32_t KH = KERNEL_SIZE;
  const int32_t KW = KERNEL_SIZE;
  const int32_t KD = KERNEL_SIZE;

  CIM_PRECISION matA[N][C][H][W][D];
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          for (int d = 0; d < D; ++d) {
            matA[n][c][h][w][d] = h;
          }
        }
      }
    }
  }

  CIM_PRECISION matB[K][C][KH][KW][KD];
  for (int k = 0; k < K; ++k) {
    for (int c = 0; c < C; ++c) {
      for (int kh = 0; kh < KH; ++kh) {
        for (int kw = 0; kw < KW; ++kw) {
          for (int kd = 0; kd < KD; ++kd) {
            matB[k][c][kh][kw][kd] = 1;
          }
        }
      }
    }
  }

  CIM_PRECISION matC[N][K][H][W][D];
  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K; ++k) {
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          for (int d = 0; d < D; ++d) {
            matC[n][k][h][w][d] = 0;
          }
        }
      }
    }
  }

  memref::MemRef<CIM_PRECISION, rank> A((CIM_PRECISION *)matA, {N, C, H, W, D});
  memref::MemRef<CIM_PRECISION, rank> B((CIM_PRECISION *)matB,
                                        {K, C, KH, KW, KD});
  memref::MemRef<CIM_PRECISION, rank> memC((CIM_PRECISION *)matC,
                                           {N, K, H, W, D});

  std::cout << "A input:\n";
  utility::printTensor(A);

  std::cout << "B input:\n";
  utility::printTensor(B);

  simulator_mark_start();
#ifdef BENCH_BUILD_ARM
  const int paddingRows = KH - 1;
  const int paddingCols = KW - 1;
  const int paddingDepth = KD - 1;
  CIM_PRECISION matPaddedA[N][K][H + paddingRows][W + paddingCols]
                          [D + paddingDepth] = {0};
  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K; ++k) {
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          for (int d = 0; d < D; ++d) {
            matPaddedA[n][k][h + paddingRows / 2 + (paddingRows % 2)]
                      [w + paddingCols / 2 + (paddingCols % 2)]
                      [d + paddingDepth / 2 + (paddingDepth % 2)] =
                          matA[n][k][h][w][d];
          }
        }
      }
    }
  }
  memref::MemRef<CIM_PRECISION, rank> paddedA(
      (CIM_PRECISION *)matPaddedA,
      {N, C, H + paddingRows, W + paddingCols, D + paddingDepth});
  _mlir_ciface_conv3d(&paddedA.memRefDesc, &B.memRefDesc, &memC.memRefDesc);
#else
  _mlir_ciface_conv3d(&A.memRefDesc, &B.memRefDesc, &memC.memRefDesc);
#endif // BENCH_ARM_BUILD
  simulator_mark_end();

  std::cout << "C output:\n";
  utility::printTensor(memC);

  simulator_terminate();
}
