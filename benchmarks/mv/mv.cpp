#include "mlir_interface/memref/memref.hpp"
#include "simulator_interface/cim_sim.h"
#include "utility/utility.hpp"

#include <array>
#include <cstdint>

#define CIM_PRECISION int8_t

#ifndef DIM_SIZE
#define DIM_SIZE 3
#endif // DIM_SIZE

static const size_t rankA = 2;
static const size_t rankB = 1;
static const size_t rankC = 1;

// Functions generated from TC
extern "C" {
void _mlir_ciface_mv(memref::MemRefDescriptor<CIM_PRECISION, rankA> *A,
                     memref::MemRefDescriptor<CIM_PRECISION, rankB> *B,
                     memref::MemRefDescriptor<CIM_PRECISION, rankC> *C);
}

int main() {
  simulator_init();

  // Matrix-vector multiplication
  // C[N] =  A[N][K] * B[K]
  const int32_t N = DIM_SIZE;
  const int32_t K = DIM_SIZE;

  std::array<int64_t, rankA> aDim{N, K};
  std::array<int64_t, rankB> bDim{K};
  std::array<int64_t, rankC> cDim{N};

  const int aSize = utility::tensorSize<>(aDim);
  const int bSize = utility::tensorSize<>(bDim);
  const int cSize = utility::tensorSize<>(cDim);

  CIM_PRECISION matA[aSize];
  for (int i = 0; i < aSize; ++i) {
    matA[i] = i;
  }

  CIM_PRECISION matB[bSize];
  for (int i = 0; i < bSize; ++i) {
    matB[i] = i;
  }

  CIM_PRECISION matC[cSize];
  for (int i = 0; i < cSize; ++i) {
    matC[i] = 0;
  }

  memref::MemRef<CIM_PRECISION, rankA> A((CIM_PRECISION *)matA, aDim);
  memref::MemRef<CIM_PRECISION, rankB> B((CIM_PRECISION *)matB, bDim);
  memref::MemRef<CIM_PRECISION, rankC> C((CIM_PRECISION *)matC, cDim);

  std::cout << "A Matrix:\n";
#ifdef BENCH_PRINT
  utility::printTensor(A);
#else
  printf("Rows: %d Cols: %d\n", static_cast<int>(A.memRefDesc.sizes[0]),
         static_cast<int>(A.memRefDesc.sizes[1]));
#endif // BENCH_PRINT

  std::cout << "B Vector:\n";
#ifdef BENCH_PRINT
  utility::printTensor(B);
#else
  printf("Rows: %d\n", static_cast<int>(B.memRefDesc.sizes[0]));
#endif // BENCH_PRINT

  simulator_mark_start();
  _mlir_ciface_mv(&A.memRefDesc, &B.memRefDesc, &C.memRefDesc);
  simulator_mark_end();

  std::cout << "GEMV result:\n";
#ifdef BENCH_PRINT
  utility::printTensor(C);
#else
  printf("Rows: %d\n", static_cast<int>(C.memRefDesc.sizes[0]));
#endif // BENCH_PRINT

  simulator_terminate();
}
