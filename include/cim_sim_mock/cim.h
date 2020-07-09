#ifndef _CIM_SIM_MOCK__CIM_H_
#define _CIM_SIM_MOCK__CIM_H_

#ifdef __cplusplus
#include <cstdint>
extern "C" {
#else
#include <stdint.h>
#endif

/**
 * Wait for computation completion on the specified tile
 */
void cim_await(uint32_t cim_id);

/**
 * Store B matrix on CIM crossbar
 */
void cim_configure_crossbar(uint32_t cim_id, uint8_t **B, uint8_t *bias,
                            uint32_t N, uint32_t K);

/**
 * Perform GEMM on CIM
 * C[M][N] =  A[M][K] * B_XBar[K][N]
 */
void cim_gemm(uint32_t cim_id, uint8_t *A, int M, int N, int K, int K_lda,
              int x_pos, int y_pos, uint8_t *C, int N_ldc);

/**
 * Perform GEVM on CIM
 * C[N] =  A[K] * B_XBar[K][N]
 */
void cim_gevm(uint32_t cim_id, uint8_t *A, int K, int N, int x_pos, int y_pos,
              uint8_t *C);

#ifdef __cplusplus
}
#endif

#endif /* _CIM_SIM_MOCK__CIM_H_ */