#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <cmath>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
const int WARPSIZE = 32; // warpSize is not constexpr

#define WARMUP_ITERS 10
#define BENCH_ITERS 50

namespace wt {
template <const int BM, const int BN, const int BK,
          const int rowStrideB, const int elementsPerThread, const int NUM_THREADS>
__device__ void loadFromGmem(int N, int K, const float *__restrict__ A, 
                             const float *__restrict__ B,
                             float *__restrict__ As, float *__restrict__ Bs, 
                             const int threadID,
                             int innerRowB, int innerColB) {

  for (int i = 0; i < elementsPerThread; i++) {
    int index = threadID + i * NUM_THREADS;

    int row = index / BK;
    int col = index % BK;

    As[col * BM + row] = A[(row * K) + col];
  }

  for (unsigned int offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
    reinterpret_cast<float4 *>(
        &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
        reinterpret_cast<const float4 *>(
            &B[(innerRowB + offset) * N + innerColB * 4])[0];
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
    for (unsigned int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      for (unsigned int i = 0; i < TM; ++i) {
        regM[wSubRowIdx * TM + i] =
            As[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM +
               threadRowInWarp * TM + i];
      }
    }
    for (unsigned int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      for (unsigned int i = 0; i < TN; ++i) {
        regN[wSubColIdx * TN + i] =
            Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN +
               threadColInWarp * TN + i];
      }
    }

    // execute warptile matmul
    for (unsigned int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      for (unsigned int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
        // calculate per-thread results
        for (unsigned int resIdxM = 0; resIdxM < TM; ++resIdxM) {
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
    sgemmWarptiling(int M, int N, int K, float alpha, 
                    const float *__restrict__ A, 
                    const float *__restrict__ B,
                    float beta, 
                    float *__restrict__ C) {
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
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  // Move C_ptr to warp's output tile
  C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

  // calculating the indices that this thread will load into SMEM
  // we'll load 128bit / 32bit = 4 elements per thread at each step
  const unsigned int innerRowB = threadIdx.x / (BN / 4);
  const unsigned int innerColB = threadIdx.x % (BN / 4);
  constexpr unsigned int rowStrideB = NUM_THREADS / (BN / 4);

  // allocate thread-local cache for results in registerfile
  float threadResults[WMITER * TM * WNITER * TN] = {0.0};
  // we cache into registers on the warptile level
  float regM[WMITER * TM] = {0.0};
  float regN[WNITER * TN] = {0.0};

  const int totalElements = BM * BK;
  const int elementsPerThread = (totalElements + NUM_THREADS - 1) / NUM_THREADS;

  // outer-most loop over block tiles
  for (unsigned int bkIdx = 0; bkIdx < K; bkIdx += BK) {
    wt::loadFromGmem<BM, BN, BK, rowStrideB, elementsPerThread, NUM_THREADS>(
        N, K, A, B, As, Bs, threadIdx.x, innerRowB, innerColB);
    __syncthreads();
    wt::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM,
                        TN>(regM, regN, threadResults, As, Bs, warpRow, warpCol,
                            threadRowInWarp, threadColInWarp);
    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down
    __syncthreads();
  }

  // write out the results
  for (unsigned int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
    for (unsigned int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      // move C pointer to current warp subtile
      float *C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
      for (unsigned int resIdxM = 0; resIdxM < TM; resIdxM += 1) {
        for (unsigned int resIdxN = 0; resIdxN < TN; resIdxN += 4) {
          // load C vector into registers
          float4 tmp = reinterpret_cast<float4 *>(
              &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                         threadColInWarp * TN + resIdxN])[0];
          // perform GEMM update in reg
          const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                        wSubColIdx * TN + resIdxN;
          tmp.x = alpha * threadResults[i + 0] + beta * tmp.x;
          tmp.y = alpha * threadResults[i + 1] + beta * tmp.y;
          tmp.z = alpha * threadResults[i + 2] + beta * tmp.z;
          tmp.w = alpha * threadResults[i + 3] + beta * tmp.w;
          // write back
          reinterpret_cast<float4 *>(
              &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                         threadColInWarp * TN + resIdxN])[0] = tmp;
        }
      }
    }
  }
}

float *compareKernelAndCUBLAS(int m, int k, int n)
{
    size_t sizeA = m * k * sizeof(float);
    size_t sizeB = k * n * sizeof(float);
    size_t sizeC = m * n * sizeof(float);

    float *hA = new float[m * k];
    float *hB = new float[k * n];
    float *hC = new float[m * n];
    float *hCRef = new float[m * n];

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(1.0f, 1000.0f);

    for (int i = 0; i < m * k; i++) hA[i] = dist(gen);
    for (int i = 0; i < k * n; i++) hB[i] = dist(gen);

    float *dA, *dB, *dC;
    cudaMalloc(&dA, sizeA);
    cudaMalloc(&dB, sizeB);
    cudaMalloc(&dC, sizeC);

    cudaMemcpy(dA, hA, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeB, cudaMemcpyHostToDevice);

    // Custom kernel setup
    const unsigned int WARPSIZE = 32;

    /* GOOD PARAMETERS */
    /*
        const unsigned int K10_NUM_THREADS = 32;
        const unsigned int K10_BM = 32;
        const unsigned int K10_BN = 128;
        const unsigned int K10_BK = 16;
        const unsigned int K10_WM = 32;
        const unsigned int K10_WN = 128;
        const unsigned int K10_WNITER = 4;
        const unsigned int K10_TM = 2;
        const unsigned int K10_TN = 4;
    */

    const unsigned int K10_NUM_THREADS = 32;
    const unsigned int K10_BM = 32;
    const unsigned int K10_BN = 128;
    const unsigned int K10_BK = 16;
    const unsigned int K10_WM = 32;
    const unsigned int K10_WN = 128;
    const unsigned int K10_WNITER = 4;
    const unsigned int K10_TM = 2;
    const unsigned int K10_TN = 4;

    dim3 blockDim(K10_NUM_THREADS);

    constexpr unsigned int NUM_WARPS = K10_NUM_THREADS / 32;

    // warptile in threadblocktile
    static_assert((K10_BN % K10_WN == 0) && (K10_BM % K10_WM == 0), "");
    static_assert((K10_BN / K10_WN) * (K10_BM / K10_WM) == NUM_WARPS, "");

    // threads in warpsubtile
    static_assert((K10_WM * K10_WN) % (WARPSIZE * K10_TM * K10_TN * K10_WNITER) ==
                    0, "");
    constexpr unsigned int K10_WMITER =
        (K10_WM * K10_WN) / (WARPSIZE * K10_TM * K10_TN * K10_WNITER);
    // warpsubtile in warptile
    static_assert((K10_WM % K10_WMITER == 0) && (K10_WN % K10_WNITER == 0), "");

    static_assert((K10_NUM_THREADS * 4) % K10_BK == 0,
                    "NUM_THREADS*4 must be multiple of K9_BK to avoid quantization "
                    "issues during GMEM->SMEM tiling (loading only parts of the "
                    "final row of Bs during each iteraion)");
    static_assert((K10_NUM_THREADS * 4) % K10_BN == 0,
                    "NUM_THREADS*4 must be multiple of K9_BN to avoid quantization "
                    "issues during GMEM->SMEM tiling (loading only parts of the "
                    "final row of As during each iteration)");
    static_assert(K10_BN % (16 * K10_TN) == 0,
                    "BN must be a multiple of 16*TN to avoid quantization effects");
    static_assert(K10_BM % (16 * K10_TM) == 0,
                    "BM must be a multiple of 16*TM to avoid quantization effects");
    static_assert((K10_BM * K10_BK) % (4 * K10_NUM_THREADS) == 0,
                    "BM*BK must be a multiple of 4*256 to vectorize loads");
    static_assert((K10_BN * K10_BK) % (4 * K10_NUM_THREADS) == 0,
                    "BN*BK must be a multiple of 4*256 to vectorize loads");

    dim3 gridDim(CEIL_DIV(n, K10_BN), CEIL_DIV(m, K10_BM));

    // cuBLAS setup
    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // Warm up
    for (int i = 0; i < WARMUP_ITERS; i++) {
        sgemmWarptiling<K10_BM, K10_BN, K10_BK, K10_WM, K10_WN, K10_WNITER, K10_TM,
                  K10_TN, K10_NUM_THREADS>
          <<<gridDim, blockDim>>>(m, n, k, alpha, dA, dB, beta, dC);
    }

    for (int i = 0; i < WARMUP_ITERS; i++) {
        cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    n, m, k,
                    &alpha,
                    dB, n,
                    dA, k,
                    &beta,
                    dC, n);
    }

    cudaDeviceSynchronize();

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // Benchmark custom kernel
    cudaEventRecord(start);
    for (int i = 0; i < BENCH_ITERS; i++) {
        sgemmWarptiling<K10_BM, K10_BN, K10_BK, K10_WM, K10_WN, K10_WNITER, K10_TM,
                  K10_TN, K10_NUM_THREADS>
          <<<gridDim, blockDim>>>(m, n, k, alpha, dA, dB, beta, dC);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float kernelMs;
    cudaEventElapsedTime(&kernelMs, start, end);
    kernelMs /= BENCH_ITERS;

    cudaMemcpy(hC, dC, sizeC, cudaMemcpyDeviceToHost);

    // Benchmark cuBLAS
    cudaEventRecord(start);
    for (int i = 0; i < BENCH_ITERS; i++) {
        cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    n, m, k,
                    &alpha,
                    dB, n,
                    dA, k,
                    &beta,
                    dC, n);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float cublasMs;
    cudaEventElapsedTime(&cublasMs, start, end);
    cublasMs /= BENCH_ITERS;

    cudaMemcpy(hCRef, dC, sizeC, cudaMemcpyDeviceToHost);

    // Validation
    int errors = 0;
    const float relTol = 1e-5f;
    const float absTol = 1e-3f;

    for (int i = 0; i < m * n; i++) {
        float diff = fabs(hCRef[i] - hC[i]);
        float maxVal = fmax(fabs(hCRef[i]), fabs(hC[i]));
        if (maxVal < 1.0f) maxVal = 1.0f;

        if (diff > absTol && diff / maxVal > relTol) {
            if (++errors <= 10) {
                std::cout << "Mismatch at " << i << ": "
                          << hC[i] << " vs " << hCRef[i] << "\n";
            }
        }
    }

    if (errors == 0) {
        std::cout << "Results match!\n";
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cublasDestroy(handle);

    delete[] hA;
    delete[] hB;
    delete[] hC;
    delete[] hCRef;

    return new float[2]{kernelMs, cublasMs};
}

int main()
{
    int m = 32;
    int k = 4096;
    int n = 12288;

    float *runtimes = compareKernelAndCUBLAS(m, k, n);

    std::cout << "Custom kernel avg ms: " << runtimes[0] << "\n";
    std::cout << "cuBLAS avg ms:        " << runtimes[1] << "\n";

    delete[] runtimes;
    return 0;
}