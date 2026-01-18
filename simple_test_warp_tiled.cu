#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <cmath>

#include "sgemm_warp_tiling_bt.cuh"

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

#define WARMUP_ITERS 1
#define BENCH_ITERS 1

float *compareKernelAndCUBLAS(int m, int k, int n)
{
    size_t sizeA = m * k * sizeof(float);
    size_t sizeBT = n * k * sizeof(float);
    size_t sizeC = m * n * sizeof(float);

    float *hA = new float[m * k];
    float *hBT = new float[n * k];
    float *hC = new float[m * n];
    float *hCRef = new float[m * n];

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(1.0f, 1000.0f);

    for (int i = 0; i < m * k; i++) hA[i] = dist(gen);
    for (int i = 0; i < n * k; i++) hBT[i] = dist(gen);

    float *dA, *dBT, *dC;
    cudaMalloc(&dA, sizeA);
    cudaMalloc(&dBT, sizeBT);
    cudaMalloc(&dC, sizeC);

    cudaMemcpy(dA, hA, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(dBT, hBT, sizeBT, cudaMemcpyHostToDevice);

    // Custom kernel setup
    const unsigned int WARPSIZE = 32;

    /* GOOD PARAMETERS */
    /*
      const unsigned int K10_NUM_THREADS = 256;
      const unsigned int K10_BM = 32;
      const unsigned int K10_BN = 128;
      const unsigned int K10_BK = 32;
      const unsigned int K10_WM = 16;
      const unsigned int K10_WN = 32;
      const unsigned int K10_WNITER = 2;
      const unsigned int K10_TM = 2;
      const unsigned int K10_TN = 4;
    */

    // B = 32
    const unsigned int K10_NUM_THREADS = 256;
    const unsigned int K10_BM = 32;
    const unsigned int K10_BN = 128;
    const unsigned int K10_BK = 32;
    const unsigned int K10_WM = 16;
    const unsigned int K10_WN = 32;
    const unsigned int K10_WNITER = 2;
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

    // --------------------
    // Warm-up
    // --------------------
    for (int i = 0; i < WARMUP_ITERS; i++) {
        sgemmWarptiling_BT<K10_BM, K10_BN, K10_BK, K10_WM, K10_WN, K10_WNITER, K10_TM,
                  K10_TN, K10_NUM_THREADS>
          <<<gridDim, blockDim>>>(m, n, k, dA, dBT, dC);
    }

    // Compute C^T = B^T * A^T
    for (int i = 0; i < WARMUP_ITERS; i++) {
        cublasSgemm(handle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    n, m, k,
                    &alpha,
                    dBT, k,
                    dA, k,
                    &beta,
                    dC, n);
    }

    cudaDeviceSynchronize();

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // --------------------
    // Benchmark custom kernel
    // --------------------
    cudaEventRecord(start);
    for (int i = 0; i < BENCH_ITERS; i++) {
        sgemmWarptiling_BT<K10_BM, K10_BN, K10_BK, K10_WM, K10_WN, K10_WNITER, K10_TM,
                  K10_TN, K10_NUM_THREADS>
          <<<gridDim, blockDim>>>(m, n, k, dA, dBT, dC);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float kernelMs;
    cudaEventElapsedTime(&kernelMs, start, end);
    kernelMs /= BENCH_ITERS;

    cudaMemcpy(hC, dC, sizeC, cudaMemcpyDeviceToHost);

    // --------------------
    // Benchmark cuBLAS
    // --------------------
    cudaEventRecord(start);
    for (int i = 0; i < BENCH_ITERS; i++) {
        cublasSgemm(handle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    n, m, k,
                    &alpha,
                    dBT, k,
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

    // --------------------
    // Validation
    // --------------------
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
    cudaFree(dBT);
    cudaFree(dC);
    cublasDestroy(handle);

    delete[] hA;
    delete[] hBT;
    delete[] hC;
    delete[] hCRef;

    return new float[2]{kernelMs, cublasMs};
}

int main()
{
    int m = 4;
    int k = 4096;
    int n = 12288;

    float *runtimes = compareKernelAndCUBLAS(m, k, n);

    std::cout << "Custom kernel avg ms: " << runtimes[0] << "\n";
    std::cout << "cuBLAS avg ms:        " << runtimes[1] << "\n";

    delete[] runtimes;
    return 0;
}