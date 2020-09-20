#include "cim_sim_mock/cim.h"

#include <stdio.h>

#define TEXT_SET_COLOR "\033[0;32m"
#define TEXT_RESET_COLOR "\033[0m"

#define XBAR_PRECISION uint32_t
#define NUM_CIM_TILES 16

// Use fixed crossbar precision
static XBAR_PRECISION *xbar[NUM_CIM_TILES];

void cim_await(uint32_t cim_id) {
  printf("[CIM %d Info] Computation completed\n", cim_id);
}

void cim_configure_crossbar(uint32_t cim_id, uint8_t **B, uint8_t *bias,
                            uint32_t N, uint32_t K) {
  printf(TEXT_SET_COLOR);
  printf("[CIM %d Info] Crossbar configured\n", cim_id);
  printf(TEXT_RESET_COLOR);

  xbar[cim_id] = (XBAR_PRECISION *)B;
}

void cim_gemm(uint32_t cim_id, uint8_t *A, int M, int N, int K, int K_lda,
              int x_pos, int y_pos, uint8_t *C, int N_ldc) {
  printf("[CIM %d Info] GEMM started...\n", cim_id);

  XBAR_PRECISION *matA = (XBAR_PRECISION *)A;
  XBAR_PRECISION *matC = (XBAR_PRECISION *)C;

  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      XBAR_PRECISION sum = 0;

      for (int k = 0; k < K; ++k) {
        sum += matA[m * K + k] * ((xbar[cim_id])[k * N + n]);
      }

      matC[m * N + n] = sum;
    }
  }
}

void cim_gevm(uint32_t cim_id, uint8_t *A, int K, int N, int x_pos, int y_pos,
              uint8_t *C) {
  printf("[CIM %d Info] GEVM started...\n", cim_id);

  XBAR_PRECISION *matA = (XBAR_PRECISION *)A;
  XBAR_PRECISION *matC = (XBAR_PRECISION *)C;

  for (int n = 0; n < N; ++n) {
    XBAR_PRECISION sum = 0;

    for (int k = 0; k < K; ++k) {
      sum += matA[k] * ((xbar[cim_id])[k * N + n]);
    }

    matC[n] = sum;
  }
}
