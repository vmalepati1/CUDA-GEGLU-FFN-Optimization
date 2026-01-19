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

namespace ffn {
template <const int BM, const int BN, const int BK, const int rowStrideA,
          const int rowStrideB>
__device__ void loadFromGmemFused(int N, int K, const float *__restrict__ X, 
                                  const float *__restrict__ Wu,
                                  const float *__restrict__ Wv,
                                  float *__restrict__ Xs, 
                                  float *__restrict__ Wus,
                                  float *__restrict__ Wvs,
                                  int innerRowA, int innerColA,
                                  int innerRowB, int innerColB) {
  // Load X into shared memory
  for (unsigned int offset = 0; offset <= BM - rowStrideA; offset += rowStrideA) {
    const float4 tmp = reinterpret_cast<const float4 *>(
        &X[(innerRowA + offset) * K + innerColA * 4])[0];

    Xs[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
    Xs[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
    Xs[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
    Xs[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
  }

  // Load Wu and Wv into shared memory (fused for better ILP)
  for (unsigned int offset = 0; offset <= BN - rowStrideB; offset += rowStrideB) {
    const float4 tmpWu = reinterpret_cast<const float4 *>(
        &Wu[(innerRowB + offset) * K + innerColB * 4])[0];
    
    const float4 tmpWv = reinterpret_cast<const float4 *>(
        &Wv[(innerRowB + offset) * K + innerColB * 4])[0];

    Wus[(innerColB * 4 + 0) * BN + innerRowB + offset] = tmpWu.x;
    Wus[(innerColB * 4 + 1) * BN + innerRowB + offset] = tmpWu.y;
    Wus[(innerColB * 4 + 2) * BN + innerRowB + offset] = tmpWu.z;
    Wus[(innerColB * 4 + 3) * BN + innerRowB + offset] = tmpWu.w;

    Wvs[(innerColB * 4 + 0) * BN + innerRowB + offset] = tmpWv.x;
    Wvs[(innerColB * 4 + 1) * BN + innerRowB + offset] = tmpWv.y;
    Wvs[(innerColB * 4 + 2) * BN + innerRowB + offset] = tmpWv.z;
    Wvs[(innerColB * 4 + 3) * BN + innerRowB + offset] = tmpWv.w;
  }
}

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,
          const int TM, const int TN>
__device__ void
processFromSmemFused(float *__restrict__ regM, 
                     float *__restrict__ regNWu,
                     float *__restrict__ regNWv,
                     float *__restrict__ threadResultsWu,
                     float *__restrict__ threadResultsWv,
                     const float *__restrict__ Xs,
                     const float *__restrict__ Wus,
                     const float *__restrict__ Wvs,
                     const unsigned int warpRow, const unsigned int warpCol,
                     const unsigned int threadRowInWarp, const unsigned int threadColInWarp) {
  for (unsigned int dotIdx = 0; dotIdx < BK; ++dotIdx) {
    // Populate registers for whole warptile from X (shared by both computations)
    #pragma unroll
    for (unsigned int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      #pragma unroll
      for (unsigned int i = 0; i < TM; ++i) {
        regM[wSubRowIdx * TM + i] =
            Xs[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM +
               threadRowInWarp * TM + i];
      }
    }

    // Populate registers from Wu and Wv (fused for better ILP)
    #pragma unroll
    for (unsigned int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      #pragma unroll
      for (unsigned int i = 0; i < TN; ++i) {
        regNWu[wSubColIdx * TN + i] =
            Wus[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN +
                threadColInWarp * TN + i];

        regNWv[wSubColIdx * TN + i] =
            Wvs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN +
                threadColInWarp * TN + i];
      }
    }

    // Execute warptile matmul for both X*Wu and X*Wv (fused for better ILP)
    #pragma unroll
    for (unsigned int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      #pragma unroll
      for (unsigned int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
        #pragma unroll
        for (unsigned int resIdxM = 0; resIdxM < TM; ++resIdxM) {
          #pragma unroll
          for (unsigned int resIdxN = 0; resIdxN < TN; ++resIdxN) {
            const int idx = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                           (wSubColIdx * TN) + resIdxN;
            const float m_val = regM[wSubRowIdx * TM + resIdxM];
            
            threadResultsWu[idx] += m_val * regNWu[wSubColIdx * TN + resIdxN];
            threadResultsWv[idx] += m_val * regNWv[wSubColIdx * TN + resIdxN];
          }
        }
      }
    }
  }
}

__device__ __forceinline__ float fast_gelu(float x) {
    // GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    return 0.5f * x * (1.0f + erff(x * 0.70710678118f)); 
}

} // namespace ffn

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
    sgemmWarptiling_Fused(int M, int N, int K,
                          const float *__restrict__ X, 
                          const float *__restrict__ Wu,
                          const float *__restrict__ Wv,
                          float *__restrict__ h) {
  const unsigned int cRow = blockIdx.y;
  const unsigned int cCol = blockIdx.x;

  // Placement of the warp in the threadblock tile
  const unsigned int warpIdx = threadIdx.x / WARPSIZE;
  const unsigned int warpCol = warpIdx % (BN / WN);
  const unsigned int warpRow = warpIdx / (BN / WN);

  // Size of the warp subtile
  constexpr unsigned int WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
  constexpr unsigned int WSUBM = WM / WMITER;
  constexpr unsigned int WSUBN = WN / WNITER;

  // Placement of the thread in the warp subtile
  const unsigned int threadIdxInWarp = threadIdx.x % WARPSIZE;
  const unsigned int threadColInWarp = threadIdxInWarp % (WSUBN / TN);
  const unsigned int threadRowInWarp = threadIdxInWarp / (WSUBN / TN);

  // Allocate space for the current blocktile in SMEM
  __shared__ float Xs[BM * BK];
  __shared__ float Wus[BK * BN];
  __shared__ float Wvs[BK * BN];

  // Move blocktile to beginning of X's row and Wu/Wv's column
  X += cRow * BM * K;
  Wu += cCol * BN * K;
  Wv += cCol * BN * K;

  // Move output pointers to warp's output tile
  h += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

  // Calculating the indices that this thread will load into SMEM
  const unsigned int innerRowA = threadIdx.x / (BK / 4);
  const unsigned int innerColA = threadIdx.x % (BK / 4);
  constexpr unsigned int rowStrideA = (NUM_THREADS * 4) / BK;
  const unsigned int innerRowB = threadIdx.x / (BK / 4);
  const unsigned int innerColB = threadIdx.x % (BK / 4);
  constexpr unsigned int rowStrideB = (NUM_THREADS * 4) / BK;

  // Allocate thread-local cache for results in registerfile
  float threadResultsWu[WMITER * TM * WNITER * TN] = {0.0};
  float threadResultsWv[WMITER * TM * WNITER * TN] = {0.0};

  // We cache into registers on the warptile level
  float regM[WMITER * TM] = {0.0};
  float regNWu[WNITER * TN] = {0.0};
  float regNWv[WNITER * TN] = {0.0};

  // Outer-most loop over block tiles
  for (unsigned int bkIdx = 0; bkIdx < K; bkIdx += BK) {
    ffn::loadFromGmemFused<BM, BN, BK, rowStrideA, rowStrideB>(
        N, K, X, Wu, Wv, Xs, Wus, Wvs, innerRowA, innerColA, innerRowB, innerColB);
    __syncthreads();
    
    ffn::processFromSmemFused<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM, TN>(
        regM, regNWu, regNWv, threadResultsWu, threadResultsWv, 
        Xs, Wus, Wvs, warpRow, warpCol, threadRowInWarp, threadColInWarp);
    
    X += BK;   // move BK columns to right
    Wu += BK;  // move BK columns to right
    Wv += BK;  // move BK columns to right
    __syncthreads();
  }

  // Write out the results (fused for better ILP and code reuse)
  for (unsigned int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
    for (unsigned int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      float *h_interim = h + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
        
      for (unsigned int resIdxM = 0; resIdxM < TM; resIdxM += 1) {
        for (unsigned int resIdxN = 0; resIdxN < TN; resIdxN += 4) {
            const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                          wSubColIdx * TN + resIdxN;
            const int offset = (threadRowInWarp * TM + resIdxM) * N +
                              threadColInWarp * TN + resIdxN;

            float4 u_val = reinterpret_cast<float4 *>(&threadResultsWu[i])[0];
            float4 v_val = reinterpret_cast<float4 *>(&threadResultsWv[i])[0];

            float4 h_res;
            h_res.x = ffn::fast_gelu(u_val.x) * v_val.x;
            h_res.y = ffn::fast_gelu(u_val.y) * v_val.y;
            h_res.z = ffn::fast_gelu(u_val.z) * v_val.z;
            h_res.w = ffn::fast_gelu(u_val.w) * v_val.w;

            reinterpret_cast<float4 *>(&h_interim[offset])[0] = h_res;
        }
      }
    }
  }
}

#endif /* FUSED_FFN_H */