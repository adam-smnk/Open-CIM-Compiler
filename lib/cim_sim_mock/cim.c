#include "cim_sim_mock/cim.h"

#include <stdio.h>

// Use fixed crossbar precision
static uint32_t *xbar;

void cim_await(uint32_t cim_id) {
  printf("[CIM Info] Computation completed\n");
}

void cim_configure_crossbar(uint32_t cim_id, uint8_t **B, uint8_t *bias,
                            uint32_t N, uint32_t K) {
  printf("[CIM Info] Crossbar configured\n");

  xbar = (uint32_t *)B;
}

void cim_gemm(uint32_t cim_id, uint8_t *A, int M, int N, int K, int K_lda,
              int x_pos, int y_pos, uint8_t *C, int N_ldc) {
  printf("[CIM Info] GEMM started...\n");

  uint32_t *matA = (uint32_t *)A;
  uint32_t *matC = (uint32_t *)C;

  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      uint32_t sum = 0;

      for (int k = 0; k < K; ++k) {
        sum += matA[m * K + k] * xbar[k * N + n];
      }

      matC[m * N + n] = sum;
    }
  }
}

void cim_gevm(uint32_t cim_id, uint8_t *A, int K, int N, int x_pos, int y_pos,
              uint8_t *C) {
  printf("[CIM Info] GEVM started...\n");

  uint32_t *matA = (uint32_t *)A;
  uint32_t *matC = (uint32_t *)C;

  for (int n = 0; n < N; ++n) {
    uint32_t sum = 0;

    for (int k = 0; k < K; ++k) {
      sum += matA[k] * xbar[k * N + n];
    }

    matC[n] = sum;
  }
}
