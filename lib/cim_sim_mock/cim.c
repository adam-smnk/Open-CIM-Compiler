#include "cim_sim_mock/cim.h"

#include <stdio.h>

void cim_await(uint32_t cim_id) { printf("CIM Computation completed\n"); }

void cim_configure_crossbar(uint32_t cim_id, uint8_t **B, uint8_t *bias,
                            uint32_t N, uint32_t K) {
  printf("CIM Crossbar configured\n");
}

void cim_gemm(uint32_t cim_id, uint8_t *A, int M, int N, int K, int K_lda,
              int x_pos, int y_pos, uint8_t *C, int N_ldc) {
  printf("CIM GEMM started...\n");
}

void cim_gevm(uint32_t cim_id, uint8_t *A, int K, int N, int x_pos, int y_pos,
              uint8_t *C) {
  printf("CIM GEVM started...\n");
}
