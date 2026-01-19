#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "fused_ffn.cuh"
#include "sgemm_warp_tiling_bt.cuh"

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// Helper to launch the Fused GLU Kernel
template <int BM, int BN, int BK, int WM, int WN, int WNITER, int TM, int TN, int NT>
void launch_stage1(int B, int K_in, int N_out, float* X, float* Wu, float* Wv, float* H) {
    dim3 block(NT);
    dim3 grid(CEIL_DIV(N_out, BN), CEIL_DIV(B, BM));
    sgemmWarptiling_Fused<BM, BN, BK, WM, WN, WNITER, TM, TN, NT>
        <<<grid, block>>>(B, N_out, K_in, X, Wu, Wv, H);
}

// Helper to launch the Projection Kernel (Standard Matmul)
template <int BM, int BN, int BK, int WM, int WN, int WNITER, int TM, int TN, int NT>
void launch_stage2(int B, int K_in, int N_out, float* H, float* Wo, float* Y) {
    dim3 block(NT);
    dim3 grid(CEIL_DIV(N_out, BN), CEIL_DIV(B, BM));
    sgemmWarptiling_BT<BM, BN, BK, WM, WN, WNITER, TM, TN, NT>
        <<<grid, block>>>(B, N_out, K_in, H, Wo, Y);
}

void benchmark(int B) {
    const int hidden_size = 4096;
    const int intermediate_size = 12288;

    // Allocate Device Memory
    float *dX, *dWu, *dWv, *dWo, *dH, *dY;
    cudaMalloc(&dX, B * hidden_size * sizeof(float));
    cudaMalloc(&dWu, intermediate_size * hidden_size * sizeof(float));
    cudaMalloc(&dWv, intermediate_size * hidden_size * sizeof(float));
    cudaMalloc(&dWo, hidden_size * intermediate_size * sizeof(float));
    cudaMalloc(&dH, B * intermediate_size * sizeof(float)); // Intermediate buffer
    cudaMalloc(&dY, B * hidden_size * sizeof(float));      // Final output

    // (Omitted: Initialize dX, dWu, dWv, dWo with random data)

    // Events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up
    for(int i=0; i<5; ++i) {
        launch_stage1<32, 128, 32, 16, 32, 2, 2, 4, 256>(B, hidden_size, intermediate_size, dX, dWu, dWv, dH);
        launch_stage2<32, 128, 32, 16, 32, 2, 2, 4, 256>(B, intermediate_size, hidden_size, dH, dWo, dY);
    }

    cudaDeviceSynchronize();
    cudaEventRecord(start);

    // --- EXECUTION ---
    // Stage 1: H = GELU(X @ Wu^T) * (X @ Wv^T)
    if (B <= 32) {
        launch_stage1<32, 128, 32, 16, 32, 2, 2, 4, 256>(B, hidden_size, intermediate_size, dX, dWu, dWv, dH);
    } else {
        launch_stage1<128, 128, 16, 64, 64, 4, 8, 4, 128>(B, hidden_size, intermediate_size, dX, dWu, dWv, dH);
    }

    // Stage 2: Y = H @ Wo^T
    if (B <= 32) {
        launch_stage2<32, 128, 32, 16, 32, 2, 2, 4, 256>(B, intermediate_size, hidden_size, dH, dWo, dY);
    } else {
        launch_stage2<128, 128, 16, 64, 64, 4, 8, 4, 128>(B, intermediate_size, hidden_size, dH, dWo, dY);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Batch %3d: %.3f ms\n", B, milliseconds);

    // Cleanup
    cudaFree(dX); cudaFree(dWu); cudaFree(dWv); cudaFree(dWo); cudaFree(dH); cudaFree(dY);
}

int main() {
    std::vector<int> batch_sizes = {4, 8, 16, 32, 64, 128};
    
    std::cout << "Device: CUDA" << std::endl;
    for (int B : batch_sizes) {
        benchmark(B);
    }
    
    return 0;
}