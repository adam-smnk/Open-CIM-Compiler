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
void _mlir_ciface_group_convolution(
    memref::MemRefDescriptor<CIM_PRECISION, 5> *I,
    memref::MemRefDescriptor<CIM_PRECISION, 5> *W1,
    memref::MemRefDescriptor<CIM_PRECISION, 1> *B,
    memref::MemRefDescriptor<CIM_PRECISION, 5> *C);
}

int main() {
  simulator_init();

  // Group Conv2D
  const size_t rank = 4;
  const int32_t N = 1;
  const int32_t F = 3;
  const int32_t C = 1;
  const int32_t G = 3;
  const int32_t H = DIM_SIZE;
  const int32_t W = DIM_SIZE;
  const int32_t KH = KERNEL_SIZE;
  const int32_t KW = KERNEL_SIZE;
  const int32_t M = 1;

  std::array<int64_t, 5> iDim = {N, G, C, H, W};
  std::array<int64_t, 5> w1Dim = {G, F, C, KH, KW};
  std::array<int64_t, 1> bDim = {M};
  std::array<int64_t, 5> oDim = {N, G, F, H, W};

  const int iSize = utility::tensorSize<>(iDim);
  const int w1Size = utility::tensorSize<>(w1Dim);
  const int bSize = utility::tensorSize<>(bDim);
  const int oSize = utility::tensorSize<>(oDim);

  auto bufI = new CIM_PRECISION[N][G][C][H][W];
  for (int n = 0; n < N; ++n) {
    for (int g = 0; g < G; ++g) {
      for (int c = 0; c < C; ++c) {
        for (int h = 0; h < H; ++h) {
          for (int w = 0; w < W; ++w) {
            bufI[n][g][c][h][w] = h;
          }
        }
      }
    }
  }

  auto bufW1 = new CIM_PRECISION[w1Size];
  for (int i = 0; i < w1Size; ++i) {
    bufW1[i] = 1;
  }

  auto bufB = new CIM_PRECISION[bSize];
  for (int i = 0; i < bSize; ++i) {
    bufB[i] = 2;
  }

  auto bufO = new CIM_PRECISION[oSize];
  for (int i = 0; i < oSize; ++i) {
    bufO[i] = 0;
  }

  memref::MemRef<CIM_PRECISION, 5> I((CIM_PRECISION *)bufI, iDim);
  memref::MemRef<CIM_PRECISION, 5> W1((CIM_PRECISION *)bufW1, w1Dim);
  memref::MemRef<CIM_PRECISION, 1> B((CIM_PRECISION *)bufB, bDim);
  memref::MemRef<CIM_PRECISION, 5> O((CIM_PRECISION *)bufO, oDim);

  std::cout << "Inputs I:\n";
#ifdef BENCH_PRINT
  utility::printTensor(I);
#else
  utility::printDimensions(I);
#endif // BENCH_PRINT

  std::cout << "Weights W1:\n";
#ifdef BENCH_PRINT
  utility::printTensor(W1);
#else
  utility::printDimensions(W1);
#endif // BENCH_PRINT

  std::cout << "Biases B:\n";
#ifdef BENCH_PRINT
  utility::printTensor(B);
#else
  utility::printDimensions(B);
#endif // BENCH_PRINT

  simulator_mark_start();
  /**
   * The group convolution will not get detected for CIM offloading
   * so the padding has to always be added manually.
   */
  const int paddingRows = KH - 1;
  const int paddingCols = KW - 1;
  CIM_PRECISION bufPaddedI[N][G][C][H + paddingRows][W + paddingCols] = {0};

  std::array<int64_t, 5> paddedIDim = {N, G, C, H + paddingRows,
                                       W + paddingCols};
  const int paddedISize = utility::tensorSize<>(paddedIDim);
  for (int i = 0; i < paddedISize; ++i) {
    ((CIM_PRECISION *)bufPaddedI)[i] = 0;
  }

  for (int n = 0; n < N; ++n) {
    for (int g = 0; g < G; ++g) {
      for (int c = 0; c < C; ++c) {
        for (int h = 0; h < H; ++h) {
          for (int w = 0; w < W; ++w) {
            bufPaddedI[n][g][c][h + paddingRows / 2 + (paddingRows % 2)]
                      [w + paddingCols / 2 + (paddingCols % 2)] =
                          bufI[n][g][c][h][w];
          }
        }
      }
    }
  }

  memref::MemRef<CIM_PRECISION, 5> paddedI((CIM_PRECISION *)bufPaddedI,
                                           paddedIDim);
  _mlir_ciface_group_convolution(&paddedI.memRefDesc, &W1.memRefDesc,
                                 &B.memRefDesc, &O.memRefDesc);
  simulator_mark_end();

  std::cout << "Output O:\n";
#ifdef BENCH_PRINT
  utility::printTensor(O);
#else
  utility::printDimensions(O);
#endif // BENCH_PRINT

  delete[] bufI;
  delete[] bufW1;
  delete[] bufB;
  delete[] bufO;

  simulator_terminate();
}
