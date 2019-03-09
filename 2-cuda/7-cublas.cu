#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <limits.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cublas_v2.h>
#include "gpu_err_check.h"

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

const int MM = 10;
const int NN = MM * MM;
const int N = 5000000;
const bool DEBUG = 0;

//////////////////////////////////////////////////
/// DEVICE functions
//////////////////////////////////////////////////



//////////////////////////////////////////////////
/// HOST and DEVICE functions
//////////////////////////////////////////////////

__host__ __device__ double rollingAverage(double cur_avg, double new_sample, int cur_n){
    // NN is the number of data points prior to new_sample being added
    cur_avg = (new_sample + ((double) cur_n) * cur_avg)/(cur_n+1);
    return cur_avg;
}

__host__ __device__ void print_grid(float* mat, int numthreads){
    printf("\n\n---------\n");
    for (int i=0; i<NN*numthreads; i+=NN){
        for (int j=0; j<MM; j++){
            for (int k=0; k<MM; k++) {
                printf("%2.0f ", mat[i + j + k*MM]);
            }
            printf("\n");
        }
        printf("---------\n");
    }
}


////////////////////////////////////////////////////
///// KERNELS
////////////////////////////////////////////////////

__global__ void setup_kernel(curandState *state, unsigned long long seed){
    int tid = threadIdx.x+blockDim.x*blockIdx.x;
    curand_init(seed * (tid+1), tid*2, 2, &state[tid]);
}


__global__ void setup_draws(float* draws){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // populate sequential numbers
    for(int i=0; i < NN; i++){
        draws[IDX2C(i,tid,NN)] = (float) i;
    }
}


__global__ void init_draws(curandState* state, float* draws){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int random_number;                // variable to hold random number
    int tmp;                          // variable to hold element for swapping
    int j;                            // variable for finding index to swap with

    // The arrays were generated all at once,
    // thus each thread has its own special location.

    curandState localState = state[tid];

    for(int i = MM*MM-1; i>=0; i--){
        random_number = (int) truncf( curand_uniform(&localState)*100000 );  // generate random number and make it big
        j = (random_number) % (i+1);  // get the index of a number to swap with that is less than i
        // perform the swap
        tmp = draws[IDX2C(j, tid, NN)];
        draws[IDX2C(j, tid, NN)] = draws[IDX2C(i, tid, NN)];
        draws[IDX2C(i, tid, NN)] = tmp;
    }
}


__global__ void draw_stick(float* grid, float* draws, const int iteration){
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    grid[ tid*NN + (int) draws[tid * NN +iteration] ] = 1.0;
}


__global__ void zero_memory(float* grid, int* done_flag, float* result){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    done_flag[tid] = 0;
    for (int i=0; i<NN; i++){
        grid[NN*tid + i] = 0;
    }
    for (int i=0; i<2*MM; i++){
        result[tid] = 0;
    }
}

__global__ void zero_average(float* average){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    average[tid] = 0;
}


__global__ void check_win(float* result, float* average, int iteration, int n, int* done_flag, float* grid){
    // each thread has 2 * MM entries to check.
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i=0; i<2*MM; i++){
        if (result[IDX2C(i, tid, 2*MM)] == MM && !done_flag[tid]){
            average[tid] = rollingAverage(average[tid], (float) iteration, n+1);
            done_flag[tid] = 1;
//            printf("Average: %f\n", average[tid]);
//            print_grid(grid, 2);
        }
    }
}

//__global__ void reduce_average(float* average){
//    extern __shared__ float s_average[];
//    int tid = blockIdx.x * blockDim.x + threadIdx.x;
//    s_average[threadIdx.x] = average[tid];  // load into shared memory
//    __syncthreads();
//
//    for (int i=0; i<blockDim.x; i*=2){
//        if(threadIdx.x % (2*i) == 0){
//            s_average[threadIdx.x] = rollingAverage(s_average[threadIdx.x], s_average[threadIdx.x + i], )
//        }
//    }
//}


//////////////////////////////////////////////////
/// HOST functions
//////////////////////////////////////////////////


void make_D_array(float* D){
    for (int i=0; i<MM; i++){
        for (int j=i*MM; j<i*MM+MM; j++){
            D[IDX2C(i,j,2*MM)] = 1;
        }
    }
    int offset = 0;
    for (int i = MM; i < 2 * MM; i++){
        for (int j = 0; j + offset < NN; j += MM){
            D[IDX2C(i, j+offset,2*MM)] = 1;
        }
        offset++;
    }
}


void gpu_blas_mmul(cublasHandle_t &handle, float *A, float *B, float *C, const int mm, const int nn,
                    const int kk) {
    ////////////////////////////////////////////////////////////////////////////////
    //! Multiply matrix A by matrix B and store the result in matrix C
    //! The operation that cublasSgemm performs is alpha*op(A) x op(B) + beta*C
    //! It is assumed that A and B are column-major-order format vectors. They can be
    //! converted to be CMO by using the parameter CUBLAS_OP_T
    //! @param handle   cuBLAS handle
    //! @param A        First matrix, size 2*MM x NN
    //! @param B        Second matrix, size NN x numthreads
    //! @param C        Resulting matrix, size 2*MM x numthreads
    //! @param mm        Number of rows of A
    //! @param nn        Number of columns in A / number of rows in B
    //! @param kk        number of columns in B
    ////////////////////////////////////////////////////////////////////////////////

     const float alf = 1;
     const float bet = 0;
     const float *alpha = &alf;
     const float *beta = &bet;

     // Do the actual multiplication
     cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, mm, kk, nn, alpha, A, mm, B, NN, beta, C, mm);
     cudaDeviceSynchronize();
     cudaCheckError();

}


//void print_D(float* D){
//    ////////////////////////////////////////////////////////////////////////////////
//    //! PRINT D MATRIX
//    //! @param D        Reference to matrix D to be printed.
//    ////////////////////////////////////////////////////////////////////////////////
//    for (int i = 0; i < 2 * MM; i++){
//        for (int j = i * N * NN; j < i * NN * N + NN * N; j++){
//            printf("%1.f", D[j]);
//        }
//        printf("\n");
//    }
//}

void print_matrix(float* mat, int rows, int cols, int chunk){
    printf("\n\n---------\n");
    for (int i=0; i<rows; i++){
        for (int j=0; j<cols; j++){
            printf("%2.0f ", mat[IDX2C(i,j,rows)]);
        }
        printf("\n");
        if ((i+1)*rows % chunk == 0){
            printf("---------\n");
        }
    }
    printf("\n\n");
}

int main(){
    cudaFree(0);
    clock_t begin = clock();
    srand(time(NULL));
    const int num_threads = 128;
    const int num_blocks = 900;

    ////////////////////////////////////////////////////////////////////////////////
    //! SETUP D
    // todo - change to be number of threads instead of N
    ////////////////////////////////////////////////////////////////////////////////
    float* D;
    D = (float*) calloc(NN * MM * 2, sizeof(float));
    make_D_array(D);

    float* d_D;
    cudaSafeCall( cudaMalloc((void**) &d_D, NN * MM * 2 * sizeof(float)) );
    cudaSafeCall( cudaMemcpy(d_D, D, NN * MM * 2 * sizeof(float), cudaMemcpyHostToDevice) );


    ////////////////////////////////////////////////////////////////////////////////
    //! SETUP grids, draws, results, averages
    //! There should be a grid here for every thread
    //! The grid is where the "draws" are stored
    //! "draws" is where the random inds from 0 to NN are stored
    //! "d_result" is where D x draws is stored
    //! "d_average" is where averages are stored
    ////////////////////////////////////////////////////////////////////////////////
    float* d_grid;
    float* grid;
    cudaSafeCall( cudaMalloc((void**) &d_grid, MM * MM * num_threads * num_blocks * sizeof(float)) );
    cudaDeviceSynchronize();
    grid = (float*) malloc(MM * MM * num_threads * num_blocks);

    float* d_draws;
    float* draws;
    cudaSafeCall( cudaMalloc((void**) &d_draws, NN * num_threads * num_blocks * sizeof(float)) );
    cudaDeviceSynchronize();
    draws = (float*) malloc(NN * num_threads * num_blocks * sizeof(float));

    float* d_result;
    float* result;
    cudaSafeCall( cudaMalloc((void**) &d_result, 2 * MM * num_threads * num_blocks * sizeof(float)) );
    cudaDeviceSynchronize();
    result = (float*) malloc(2 * MM * num_threads * num_blocks * sizeof(float));

    float* d_average;
    float* average;
    cudaSafeCall( cudaMalloc((void**) &d_average, MM * MM * num_threads * num_blocks * sizeof(float)) );
    cudaDeviceSynchronize();
    average = (float*) calloc(num_blocks*num_threads, sizeof(float));

    int* d_done_flag;
    cudaSafeCall( cudaMalloc((void**) &d_done_flag, num_threads * num_blocks * sizeof(int)) );


    ////////////////////////////////////////////////////////////////////////////////
    //! SETUP Random Numbers (picks)
    //! There should be a pick for each thread. Probably will be better to just
    //! iterate over these multiple times to save memory
    ////////////////////////////////////////////////////////////////////////////////
    curandState *d_state;
    cudaSafeCall( cudaMalloc(&d_state, num_threads * num_blocks * sizeof(curandState)) );
    setup_kernel<<<num_blocks, num_threads>>>(d_state, (unsigned long long) time(NULL));
    cudaSafeCall( cudaDeviceSynchronize() );
    cudaCheckError();


    //////////////////////////////////////////////////
    //! SETUP cuBLAS
    //////////////////////////////////////////////////
    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaDeviceSynchronize();
    cudaCheckError();


    //////////////////////////////////////////////////
    //! Init Draws
    //////////////////////////////////////////////////
    init_draws<<<num_blocks, num_threads>>>(d_state, d_draws);
    cudaDeviceSynchronize();
    cudaCheckError();


    // AT this point, there is a single D matrix in device memory of size 2*MM x NN * (total number of threads)
    // Also in memory is the "draws" matrix, chunked into "total number of threads" blocks of MM x MM
    // Now, we need to randomly draw one of those numbers at a time.

    int a = 2*MM;
    int b = NN;
    int c = num_blocks*num_threads;

    zero_average<<<num_blocks, num_threads>>>(d_average);
    cudaDeviceSynchronize();
    cudaCheckError();

    setup_draws<<<num_blocks, num_threads>>>(d_draws);
    cudaDeviceSynchronize();
    cudaCheckError();


    for (int n=0; n < N/num_blocks/num_threads; n++) {

        zero_memory <<< num_blocks, num_threads >>> (d_grid, d_done_flag, d_result);
        cudaDeviceSynchronize();
        cudaCheckError();

        setup_kernel<<<num_blocks, num_threads>>>(d_state, rand());
        cudaSafeCall( cudaDeviceSynchronize() );
        cudaCheckError();

        //////////////////////////////////////////////////
        //! Init Draws
        //////////////////////////////////////////////////
        init_draws<<<num_blocks, num_threads>>>(d_state, d_draws);
        cudaDeviceSynchronize();
        cudaCheckError();

        for (int i = 0; i < NN; i++) {
            draw_stick <<< num_blocks, num_threads >>> (d_grid, d_draws, i);
            cudaDeviceSynchronize();
            cudaCheckError();
            if (DEBUG) {
                cudaMemcpy(grid, d_grid, MM * MM * num_threads * num_blocks * sizeof(float), cudaMemcpyDeviceToHost);
                print_grid(grid, num_threads * num_blocks);
            }
            // Need to multiply D by matrix of grid vectors
            // size of D = 2m x NN(total number of threads)
            // size of grid_vector = NN * (total number of threads)
            // result = 2m x (total number of threads)
            gpu_blas_mmul(handle, d_D, d_grid, d_result, a, b, c);
            if (DEBUG) {
                cudaMemcpy(result, d_result, 2 * MM * num_threads * num_blocks * sizeof(float), cudaMemcpyDeviceToHost);
                print_matrix(result, 2 * MM, num_blocks * num_threads, 10000000);
            }
            check_win <<< num_blocks, num_threads >>> (d_result, d_average, i, n, d_done_flag, d_grid);
            cudaDeviceSynchronize();
            cudaCheckError();
        }
//        cudaMemcpy(average, d_average, num_blocks*num_threads*sizeof(float), cudaMemcpyDeviceToHost);
//        float end_average = 0;
//        for (int i=0; i<num_blocks * num_threads; i++){
//            end_average = rollingAverage(end_average, average[i], i);
//        }
//        fprintf(f,"%f\n",end_average);
    }
    cudaMemcpy(average, d_average, num_blocks*num_threads*sizeof(float), cudaMemcpyDeviceToHost);
//    print_matrix(average, num_blocks * num_threads, 1, 10000);
    float end_average = 0;
    for (int i=0; i<num_blocks * num_threads; i++){
        end_average = rollingAverage(end_average, average[i], i);
    }

    printf("Average: %f\n", end_average);

    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

    printf("TIME: %f\n", time_spent);

    free(D);
    cublasDestroy(handle);
    return 0;
}