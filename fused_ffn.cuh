#ifndef FUSED_FFN_H
#define FUSED_FFN_H

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <cmath>

#define WARPSIZE 32

namespace wt {
template <const int BM, const int BN, const int BK, const int rowStrideA,
          const int rowStrideB>
__device__ void loadFromGmem(int N, int K, const float *__restrict__ A, 
                             const float *__restrict__ B,
                             float *__restrict__ As, float *__restrict__ Bs, 
                             int innerRowA, int innerColA,
                             int innerRowB, int innerColB) {
  for (unsigned int offset = 0; offset <= BM - rowStrideA; offset += rowStrideA) {
    const float4 tmp = reinterpret_cast<const float4 *>(
        &A[(innerRowA + offset) * K + innerColA * 4])[0];

    // This causes a bunch of bank conflicts since we are writing along a column of As in SMEM
    As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
    As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
    As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
    As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
  }

  for (unsigned int offset = 0; offset <= BN - rowStrideB; offset += rowStrideB) {
    const float4 tmp = reinterpret_cast<const float4 *>(
        &B[(innerRowB + offset) * K + innerColB * 4])[0];

    // This causes a bunch of bank conflicts since we are writing along a column of Bs in SMEM
    Bs[(innerColB * 4 + 0) * BN + innerRowB + offset] = tmp.x;
    Bs[(innerColB * 4 + 1) * BN + innerRowB + offset] = tmp.y;
    Bs[(innerColB * 4 + 2) * BN + innerRowB + offset] = tmp.z;
    Bs[(innerColB * 4 + 3) * BN + innerRowB + offset] = tmp.w;
  }
}

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,
          const int TM, const int TN>
__device__ void
processFromSmem(float *__restrict__ regM, float *__restrict__ regN, 
                float *__restrict__ threadResults, 
                const float *__restrict__ As,
                const float *__restrict__ Bs, 
                const unsigned int warpRow, const unsigned int warpCol,
                const unsigned int threadRowInWarp, const unsigned int threadColInWarp) {
  for (unsigned int dotIdx = 0; dotIdx < BK; ++dotIdx) {
    // populate registers for whole warptile
    #pragma unroll
    for (unsigned int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {

      #pragma unroll
      for (unsigned int i = 0; i < TM; ++i) {
        regM[wSubRowIdx * TM + i] =
            As[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM +
               threadRowInWarp * TM + i];
      }
    }

    #pragma unroll
    for (unsigned int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {

      #pragma unroll
      for (unsigned int i = 0; i < TN; ++i) {
        regN[wSubColIdx * TN + i] =
            Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN +
               threadColInWarp * TN + i];
      }
    }

    // execute warptile matmul

    #pragma unroll
    for (unsigned int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {

      #pragma unroll
      for (unsigned int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
        // calculate per-thread results

        #pragma unroll
        for (unsigned int resIdxM = 0; resIdxM < TM; ++resIdxM) {

          #pragma unroll
          for (unsigned int resIdxN = 0; resIdxN < TN; ++resIdxN) {
            threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                          (wSubColIdx * TN) + resIdxN] +=
                regM[wSubRowIdx * TM + resIdxM] *
                regN[wSubColIdx * TN + resIdxN];
          }
        }
      }
    }
  }
}

} // namespace wt

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
    sgemmWarptiling_BT(int M, int N, int K,
                    const float *__restrict__ X, 
                    const float *__restrict__ Wu,
                    const float *__restrict__ Wv,
                    float *__restrict__ g,
                    float *__restrict__ v) {
  const unsigned int cRow = blockIdx.y;
  const unsigned int cCol = blockIdx.x;

  // Placement of the warp in the threadblock tile
  const unsigned int warpIdx = threadIdx.x / WARPSIZE; // the warp this thread is in
  const unsigned int warpCol = warpIdx % (BN / WN);
  const unsigned int warpRow = warpIdx / (BN / WN);

  // size of the warp subtile
  constexpr unsigned int WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
  constexpr unsigned int WSUBM = WM / WMITER; // 64/2=32
  constexpr unsigned int WSUBN = WN / WNITER; // 32/2=16

  // Placement of the thread in the warp subtile
  const unsigned int threadIdxInWarp = threadIdx.x % WARPSIZE;         // [0, 31]
  const unsigned int threadColInWarp = threadIdxInWarp % (WSUBN / TN); // i%(16/4)
  const unsigned int threadRowInWarp = threadIdxInWarp / (WSUBN / TN); // i/4

  // allocate space for the current blocktile in SMEM
  __shared__ float Xs[BM * BK];
  __shared__ float Bs[BK * BN];
  __shared__ float Bs[BK * BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN * K;
  // Move C_ptr to warp's output tile
  C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

  // calculating the indices that this thread will load into SMEM
  // we'll load 128bit / 32bit = 4 elements per thread at each step
  const unsigned int innerRowA = threadIdx.x / (BK / 4);
  const unsigned int innerColA = threadIdx.x % (BK / 4);
  constexpr unsigned int rowStrideA = (NUM_THREADS * 4) / BK;
  const unsigned int innerRowB = threadIdx.x / (BK / 4);
  const unsigned int innerColB = threadIdx.x % (BK / 4);
  constexpr unsigned int rowStrideB = (NUM_THREADS * 4) / BK;

  // allocate thread-local cache for results in registerfile
  float threadResults[WMITER * TM * WNITER * TN] = {0.0};
  // we cache into registers on the warptile level
  float regM[WMITER * TM] = {0.0};
  float regN[WNITER * TN] = {0.0};

  // outer-most loop over block tiles
  for (unsigned int bkIdx = 0; bkIdx < K; bkIdx += BK) {
    wt::loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
        N, K, A, B, As, Bs, innerRowA, innerColA, innerRowB, innerColB);
    __syncthreads();
    wt::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM,
                        TN>(regM, regN, threadResults, As, Bs, warpRow, warpCol,
                            threadRowInWarp, threadColInWarp);
    A += BK;     // move BK columns to right
    B += BK;     // move BK columns to right
    __syncthreads();
  }

  // write out the results
  for (unsigned int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
    for (unsigned int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      // move C pointer to current warp subtile
      float *C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
      for (unsigned int resIdxM = 0; resIdxM < TM; resIdxM += 1) {
        for (unsigned int resIdxN = 0; resIdxN < TN; resIdxN += 4) {

          // perform GEMM update in reg
          const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                        wSubColIdx * TN + resIdxN;

          reinterpret_cast<float4 *>(
            &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                       threadColInWarp * TN + resIdxN])[0] = 
            reinterpret_cast<float4 *>(&threadResults[i])[0];
        }
      }
    }
  }
}

#endif /* FUSED_FFN_H */