#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>

#include "fused_ffn.cuh" 

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
#define WARPSIZE 32

// CPU Reference GELU for validation
float cpu_gelu(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.70710678118f));
}

void runFusedGLU(int B, int K, int N) {
    // M = Batch, K = Hidden, N = Intermediate
    int m = B;
    int k = K;
    int n = N;

    size_t sizeX = m * k * sizeof(float);
    size_t sizeW = n * k * sizeof(float); // For both Wu and Wv
    size_t sizeH = m * n * sizeof(float);

    // Host memory
    float *hX = new float[m * k];
    float *hWu = new float[n * k];
    float *hWv = new float[n * k];
    float *hH = new float[m * n];
    float *hH_ref = new float[m * n];

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (int i = 0; i < m * k; i++) hX[i] = dist(gen);
    for (int i = 0; i < n * k; i++) hWu[i] = dist(gen);
    for (int i = 0; i < n * k; i++) hWv[i] = dist(gen);

    float *dX, *dWu, *dWv, *dH;
    cudaMalloc(&dX, sizeX);
    cudaMalloc(&dWu, sizeW);
    cudaMalloc(&dWv, sizeW);
    cudaMalloc(&dH, sizeH);

    cudaMemcpy(dX, hX, sizeX, cudaMemcpyHostToDevice);
    cudaMemcpy(dWu, hWu, sizeW, cudaMemcpyHostToDevice);
    cudaMemcpy(dWv, hWv, sizeW, cudaMemcpyHostToDevice);

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
    dim3 gridDim(CEIL_DIV(n, K10_BN), CEIL_DIV(m, K10_BM));

    sgemmWarptiling_Fused<K10_BM, K10_BN, K10_BK, K10_WM, K10_WN, K10_WNITER, 
                         K10_TM, K10_TN, K10_NUM_THREADS>
        <<<gridDim, blockDim>>>(m, n, k, dX, dWu, dWv, dH);

    cudaDeviceSynchronize();

    // We compute: H = GELU(X @ Wu^T) * (X @ Wv^T)
    float *hU_interim = new float[m * n];
    float *hV_interim = new float[m * n];

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sumU = 0.0f;
            float sumV = 0.0f;
            for (int l = 0; l < k; ++l) {
                sumU += hX[i * k + l] * hWu[j * k + l];
                sumV += hX[i * k + l] * hWv[j * k + l];
            }
            hH_ref[i * n + j] = cpu_gelu(sumU) * sumV;
        }
    }

    cudaMemcpy(hH, dH, sizeH, cudaMemcpyDeviceToHost);

    // Check results
    int errors = 0;
    for (int i = 0; i < m * n; i++) {
        if (std::abs(hH[i] - hH_ref[i]) > 1e-3f) {
            errors++;
            if (errors < 5) printf("Error at %d: GPU %f, CPU %f\n", i, hH[i], hH_ref[i]);
        }
    }

    if (errors == 0) std::cout << "Success! Fused GLU results match.\n";
    else std::cout << "Found " << errors << " mismatches.\n";

    // Clean up
    cudaFree(dX); cudaFree(dWu); cudaFree(dWv); cudaFree(dH);
    delete[] hX; delete[] hWu; delete[] hWv; delete[] hH; delete[] hH_ref;
    delete[] hU_interim; delete[] hV_interim;
}

int main() {
    int B = 4;        // Batch size
    int K = 4096;     // Hidden size
    int N = 12288;    // Intermediate size (3 * 4096)

    runFusedGLU(B, K, N);
    return 0;
}