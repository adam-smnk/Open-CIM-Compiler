#include "mlir_interface/memref/memref.hpp"
#include "utility/utility.hpp"

#include <array>
#include <cstdint>
#include <iostream>

// Functions generated from TC
extern "C" {
void _mlir_ciface_conv3d(memref::MemRefDescriptor<int32_t, 5> *A,
                         memref::MemRefDescriptor<int32_t, 5> *B,
                         memref::MemRefDescriptor<int32_t, 5> *C);
}

int main() {
  // Conv3D
  // C[N][K][H][W][D] =  A[N][C][H][W][D] * B[K][C][KH][KW][KD]
  const size_t rank = 5;
  const int32_t N = 1;
  const int32_t K = 1;
  const int32_t C = 1;
  const int32_t H = 3;
  const int32_t W = 3;
  const int32_t D = 3;
  const int32_t KH = 2;
  const int32_t KW = 2;
  const int32_t KD = 2;

  int32_t matA[N][C][H][W][D];
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

  int32_t matB[K][C][KH][KW][KD];
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

  int32_t matC[N][K][H][W][D];
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

  memref::MemRef<int32_t, rank> A((int32_t *)matA, {N, C, H, W, D});
  memref::MemRef<int32_t, rank> B((int32_t *)matB, {K, C, KH, KW, KD});
  memref::MemRef<int32_t, rank> memC((int32_t *)matC, {N, K, H, W, D});

  std::cout << "A Matrix:\n";
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      std::cout << "N: " << n << " C: " << c << "\n";
      memref::MemRef<int32_t, 3> mat((int32_t *)matA[n][c], {H, W, D});
      utility::printMatrix3D(mat);
    }
  }

  std::cout << "B Matrix:\n";
  for (int k = 0; k < K; ++k) {
    for (int c = 0; c < C; ++c) {
      std::cout << "K: " << k << " C: " << c << "\n";
      memref::MemRef<int32_t, 3> mat((int32_t *)matB[k][c], {KH, KW, KD});
      utility::printMatrix3D(mat);
    }
  }

  _mlir_ciface_conv3d(&A.memRefDesc, &B.memRefDesc, &memC.memRefDesc);

  std::cout << "C Matrix:\n";
  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K; ++k) {
      std::cout << "N: " << n << " K: " << k << "\n";
      memref::MemRef<int32_t, 3> mat((int32_t *)matC[n][k], {H, W, D});
      utility::printMatrix3D(mat);
    }
  }
}
