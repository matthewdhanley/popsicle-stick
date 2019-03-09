#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include "gpu_err_check.h"
#include <float.h>

const unsigned long int MAX_OPS = 10000000000;
const int threads_per_block = 32;
const int n_blocks = 1000;

__global__ void naive(float* a, float* res) {
    int starting_point = blockIdx.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = 0; i < threads_per_block; i++) {
        res[tid] += a[starting_point + i];
    }
}


__global__ void coalesced(float* a, float* res, int offset) {
    int tid = (blockDim.x * blockIdx.x + threadIdx.x + offset) % threads_per_block;
    int starting_point = blockIdx.x * blockDim.x;
    for (int i = 0; i < threads_per_block; i++) {
        res[tid] += a[starting_point + i];
    }
}


int main(){
    cudaFree(0); // avoid spoofing profiler.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float* a = (float*) malloc(n_blocks * threads_per_block * sizeof(float));

    for (int i = 0; i < n_blocks * threads_per_block; i++){
        a[i] = 3.0;
    }

    float* d_a;
    float* d_res;


    printf("offset, naive, coalesced\n");
    for (int i=0; i<32; i++) {
        float milliseconds;
        for (int k = 0; k < 100; k++) {
            printf("%d,",i);
            cudaMalloc((void **) &d_a, n_blocks * threads_per_block * sizeof(float));
            cudaMalloc((void **) &d_res, n_blocks * threads_per_block * sizeof(float));
            cudaMemcpy(d_a, a, n_blocks * threads_per_block * sizeof(float), cudaMemcpyHostToDevice);

            cudaEventRecord(start);
            naive <<< n_blocks, threads_per_block >>> (d_a, d_res);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaDeviceSynchronize();
            milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            printf("%.7g,", milliseconds / 1000.0);

            cudaEventRecord(start);
            coalesced<<< n_blocks, threads_per_block >>> (d_a, d_res, i);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaDeviceSynchronize();
            milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            printf("%.7g\n", milliseconds / 1000.0);


            cudaFree(&d_a);
        }
    }

    return 0;
}