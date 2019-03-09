#include <cublas_v2.h>
#include "cuda_helper.h"

typedef float ftype;
extern "C"
void run(ftype *i1, ftype  *i2, ftype *o1, int d);


__global__ void p(ftype *i1, ftype *i2, ftype *o1, int d){

    cublasHandle_t h;
    cublasCreate(&h);
    ftype alpha = 1.0;
    ftype beta = 0.0;
    cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, d, d, d, &alpha, i1, d, i2, d, &beta, o1, d);
}

void run(ftype *i1, ftype  *i2, ftype *o1, int d){

    ftype *d_i1, *d_i2, *d_o1;
    int ds = d*d*sizeof(ftype);
    cudaMalloc(&d_i1, ds);
    cudaMalloc(&d_i2, ds);
    cudaMalloc(&d_o1, ds);
    cudaMemcpy(d_i1, i1, ds, cudaMemcpyHostToDevice);
    cudaMemcpy(d_i2, i2, ds, cudaMemcpyHostToDevice);
    p<<<1,1>>>(d_i1, d_i2, d_o1, d);
    cudaMemcpy(o1, d_o1, ds, cudaMemcpyDeviceToHost);
    cudaFree(d_i1);
    cudaFree(d_i2);
    cudaFree(d_o1);
}