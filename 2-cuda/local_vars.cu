#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include "gpu_err_check.h"
#include <float.h>

const unsigned long int MAX_OPS = 10000000000;
const int threads_per_block = 512;
const int n_blocks = 1000;

__global__ void baseline(unsigned long int n_ops) {
    unsigned long int num_iterations = n_ops / threads_per_block / n_blocks / 2;
    float res;
    for (unsigned long int i = 0; i < num_iterations; i++) {
        res = 3.0 * 4.0 + 6.0;
    }
}

__global__ void global(unsigned long int n_ops, float* a, float* b, float* c, float* res) {
    unsigned long int num_iterations = n_ops / threads_per_block / n_blocks / 2;
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    for (unsigned long int i = 0; i < num_iterations; i++) {
        res[tid] += a[tid] * b[tid] + c[tid];
    }
}


__global__ void local(unsigned long int n_ops, float* a, float* b, float* c, float* res) {
    unsigned long int num_iterations = n_ops / threads_per_block / n_blocks / 2;
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    float local_a = a[tid];
    float local_b = b[tid];
    float local_c = c[tid];
    float local_res = 0.0;
    for (unsigned long int i = 0; i < num_iterations; i++) {
        local_res += local_a * local_b + local_c;
    }
    res[tid] = local_res;
}


int main(){
    cudaFree(0); // avoid spoofing profiler.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float* a = (float*) malloc(n_blocks * threads_per_block * sizeof(float));
    float* b = (float*) malloc(n_blocks * threads_per_block * sizeof(float));
    float* c = (float*) malloc(n_blocks * threads_per_block * sizeof(float));

    for (int i = 0; i < n_blocks * threads_per_block; i++){
        a[i] = 3.0;
        b[i] = 4.0;
        c[i] = 6.0;
    }

    float* d_a;
    float* d_b;
    float* d_c;
    float* d_res;


    printf("n_ops, global, local, baseline\n");
    float milliseconds;
    for (unsigned long int i = 1; i <MAX_OPS; i*=2 ) {
        for (int k=0; k<10; k++) {
            printf("%lu,", i);

            cudaEventRecord(start);
            baseline<<<n_blocks, threads_per_block>>>(i);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaDeviceSynchronize();
            milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            printf("%.7g,", milliseconds / 1000.0);


            cudaMalloc((void**) &d_a, n_blocks * threads_per_block * sizeof(float));
            cudaMalloc((void**) &d_b, n_blocks * threads_per_block * sizeof(float));
            cudaMalloc((void**) &d_c, n_blocks * threads_per_block * sizeof(float));
            cudaMalloc((void**) &d_res, n_blocks * threads_per_block * sizeof(float));
            cudaMemcpy(d_a, a, n_blocks * threads_per_block * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_b, b, n_blocks * threads_per_block * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_c, c, n_blocks * threads_per_block * sizeof(float), cudaMemcpyHostToDevice);


            cudaEventRecord(start);
            global<<< n_blocks, threads_per_block >>> (i, d_a, d_b, d_c, d_res);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaDeviceSynchronize();
            milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            printf("%.7g,", milliseconds / 1000.0);

            cudaEventRecord(start);
            local<<< n_blocks, threads_per_block >>> (i, d_a, d_b, d_c, d_res);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaDeviceSynchronize();
            milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            printf("%.7g\n", milliseconds / 1000.0);

            cudaFree(&d_a);
            cudaFree(&d_b);
            cudaFree(&d_c);
        }
    }

    return 0;
}