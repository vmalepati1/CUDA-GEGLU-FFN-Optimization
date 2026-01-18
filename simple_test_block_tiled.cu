#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <random>
#include <cmath>

#define TILE_SIZE 32
#define WARMUP_ITERS 10
#define BENCH_ITERS 50

// C(m x n) = A(m x k) * B(k x n)
__global__ void matrixMultiplyTiled(
    const float *__restrict__ A,
    const float *__restrict__ B,
    float *__restrict__ C,
    int m, int k, int n)
{
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    float dotProduct = 0.0f;

    for (int phase = 0; phase < (k + TILE_SIZE - 1) / TILE_SIZE; phase++) {

        int rowA = blockIdx.y * TILE_SIZE + threadIdx.y;
        int colA = phase * TILE_SIZE + threadIdx.x;

        int rowB = phase * TILE_SIZE + threadIdx.y;
        int colB = blockIdx.x * TILE_SIZE + threadIdx.x;

        tileA[threadIdx.y][threadIdx.x] =
            (rowA < m && colA < k) ? A[rowA * k + colA] : 0.0f;

        tileB[threadIdx.y][threadIdx.x] =
            (rowB < k && colB < n) ? B[rowB * n + colB] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i++) {
            dotProduct += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }

        __syncthreads();
    }

    int rowC = blockIdx.y * TILE_SIZE + threadIdx.y;
    int colC = blockIdx.x * TILE_SIZE + threadIdx.x;

    if (rowC < m && colC < n) {
        C[rowC * n + colC] = dotProduct;
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

    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((n + TILE_SIZE - 1) / TILE_SIZE,
                   (m + TILE_SIZE - 1) / TILE_SIZE);

    // cuBLAS setup
    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // --------------------
    // Warm-up
    // --------------------
    for (int i = 0; i < WARMUP_ITERS; i++) {
        matrixMultiplyTiled<<<numBlocks, blockSize>>>(dA, dB, dC, m, k, n);
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

    // --------------------
    // Benchmark custom kernel
    // --------------------
    cudaEventRecord(start);
    for (int i = 0; i < BENCH_ITERS; i++) {
        matrixMultiplyTiled<<<numBlocks, blockSize>>>(dA, dB, dC, m, k, n);
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