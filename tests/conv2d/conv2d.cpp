#include "mlir_interface/memref/memref.hpp"
#include "utility/utility.hpp"

#include <array>
#include <cstdint>
#include <iostream>

// Functions generated from TC
extern "C" {
void _mlir_ciface_conv2d(memref::MemRefDescriptor<int32_t, 4> *A,
                         memref::MemRefDescriptor<int32_t, 4> *B,
                         memref::MemRefDescriptor<int32_t, 4> *C);
}

int main() {
  // Conv2D
  // C[N][K][H][W] =  A[N][C][H][W] * B[K][C][KH][KW]
  const size_t rank = 4;
  const int32_t N = 1;
  const int32_t K = 1;
  const int32_t C = 1;
  const int32_t H = 3;
  const int32_t W = 3;
  const int32_t KH = 2;
  const int32_t KW = 2;

  int32_t matA[N][C][H][W];
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          matA[n][c][h][w] = h;
        }
      }
    }
  }

  int32_t matB[K][C][KH][KW];
  for (int k = 0; k < K; ++k) {
    for (int c = 0; c < C; ++c) {
      for (int kh = 0; kh < KH; ++kh) {
        for (int kw = 0; kw < KW; ++kw) {
          matB[k][c][kh][kw] = 1;
        }
      }
    }
  }

  int32_t matC[N][K][H][W];
  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K; ++k) {
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          matC[n][k][h][w] = 0;
        }
      }
    }
  }

  memref::MemRef<int32_t, rank> A((int32_t *)matA, {N, C, H, W});
  memref::MemRef<int32_t, rank> B((int32_t *)matB, {K, C, KH, KW});
  memref::MemRef<int32_t, rank> memC((int32_t *)matC, {N, K, H, W});

  std::cout << "A Matrix:\n";
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      std::cout << "N: " << n << " C: " << c << "\n";
      memref::MemRef<int32_t, 2> mat((int32_t *)matA[n][c], {H, W});
      utility::printMatrix(mat);
    }
  }

  std::cout << "B Matrix:\n";
  for (int k = 0; k < K; ++k) {
    for (int c = 0; c < C; ++c) {
      std::cout << "K: " << k << " C: " << c << "\n";
      memref::MemRef<int32_t, 2> mat((int32_t *)matB[k][c], {KH, KW});
      utility::printMatrix(mat);
    }
  }

  _mlir_ciface_conv2d(&A.memRefDesc, &B.memRefDesc, &memC.memRefDesc);

  std::cout << "C Matrix:\n";
  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K; ++k) {
      std::cout << "N: " << n << " K: " << k << "\n";
      memref::MemRef<int32_t, 2> mat((int32_t *)matC[n][k], {H, W});
      utility::printMatrix(mat);
    }
  }
}
