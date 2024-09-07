#pragma once

#include <algorithm>
#include <cassert>
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda/barrier>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

namespace {
template <const int BM, const int BN, const int BK, const int rowStrideA,
          const int rowStrideB, typename T>
__device__ void loadFromGmem(int N, int K, float *A, float *B, float *As,
                             float *Bs, int innerRowA, int innerColA,
                             int innerRowB, int innerColB, T &barrier) {
  /*
  Your Implementation Here
  */
}

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,
          const int TM, const int TN>
__device__ void
processFromSmem(float *regM, float *regN, float *threadResults, const float *As,
                const float *Bs, const uint warpRow, const uint warpCol,
                const uint threadRowInWarp, const uint threadColInWarp) {
  /*
  Your Implementation Here
  */  
}

} // namespace

/*
 * @tparam BM The threadblock size for M dimension SMEM caching.
 * @tparam BN The threadblock size for N dimension SMEM caching.
 * @tparam BK The threadblock size for K dimension SMEM caching.
 * @tparam WM M dim of continuous tile computed by each warp
 * @tparam WN N dim of continuous tile computed by each warp
 * @tparam WMITER The number of subwarp tiling steps in M dimension.
 * @tparam WNITER The number of subwarp tiling steps in N dimension.
 * @tparam TM The per-thread tile size for M dimension.
 * @tparam TN The per-thread tile size for N dimension.
 */
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    runSgemmDoubleBuffering2(int M, int N, int K, float alpha, float *A,
                             float *B, float beta, float *C) {
  /*
  Your Implementation Here
  */ 
}