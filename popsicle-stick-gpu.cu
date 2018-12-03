//============================================================================
// Name        : popsicle-stick.cpp
// Author      : Matthew Hanley
// Version     :
// Copyright   :
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>


__global__ void setup_kernel(curandState *state){

  int idx = threadIdx.x+blockDim.x*blockIdx.x;
  curand_init(1234, idx, 0, &state[idx]);
}

__global__ void popsicle_sticks_kernel(int* number_grid, int* popsicle_sticks,
									   int grid_dim, int N,
									   curandState *state, int* good_behavior){
	int tid = blockDim.x * blockIdx.x + threadIdx.x;  // thread index
	if(tid < N){
		good_behavior[tid] = 0;
	}
	// have all the threads work together to populate the sticks, in order first.
	int stride = tid;
	int stride_step = gridDim.x;
	while(stride < grid_dim * grid_dim * N){
		popsicle_sticks[stride] = stride % (grid_dim * grid_dim);
		number_grid[stride] = 0;
		stride += stride_step;
	}
	__syncthreads();


	// now going to randomize without replacement
	stride = tid;
	int random_number;
	int tmp;
	int j;
	int start_index = tid * grid_dim * grid_dim;
	if(tid < N){
		for(int i = grid_dim*grid_dim-1; i>=0; i--){
			random_number = (int) truncf(curand_uniform(state+tid)*1000000);
			j = (random_number) % (i+1);
			tmp = popsicle_sticks[start_index + j];
			popsicle_sticks[start_index + j] = popsicle_sticks[start_index + i];
			popsicle_sticks[start_index + i] = tmp;
		}
	}
	__syncthreads();
	if (tid < N){
	int pizza_party = 0;
	int good_behavior_count = 0;

	// simulating stick drawing, a sequential process in this case.
	while (pizza_party == 0){
		int stick_drawn = popsicle_sticks[start_index + good_behavior_count];
		number_grid[start_index + stick_drawn] = 1;

		if (good_behavior_count > grid_dim){
			int col = stick_drawn % grid_dim;
			int row = stick_drawn / grid_dim;
			int row_count = 0;
			int col_count = 0;
			for(int j=0; j<grid_dim; j++){
				col_count += number_grid[start_index + j*grid_dim+col];
			}
			for(int j=row*grid_dim; j<row*grid_dim+grid_dim; j++){
				row_count += number_grid[start_index + j];
			}
			if (col_count == grid_dim || row_count == grid_dim){
				pizza_party = 1;
			}
		}
		good_behavior_count++;
	}
	good_behavior[tid] = good_behavior_count;
	}
	__syncthreads();
}


int main() {
	srand(time(NULL));

	// Number of iterations to run
	int N = 10000;

	// Grid Size
	int grid_dim = 10;

	// Init variables
	int* d_popsicle_sticks;
	int* d_good_behavior;
	int* good_behavior;
	int* d_number_grid;

	curandState *d_state;
    cudaMalloc(&d_state, sizeof(curandState));

	// allocate space on host
	popsicle_sticks_test = (int*) malloc(grid_dim * grid_dim * N * sizeof(int));
	good_behavior = (int*) malloc(N*sizeof(int));

	// allocate space on the device for lots of popsicle sticks
	cudaMalloc((void**) &d_popsicle_sticks, grid_dim * grid_dim * sizeof(int) * N);
	cudaMalloc((void**) &d_number_grid, grid_dim * grid_dim * sizeof(int) * N);
	cudaMalloc((void**) &d_good_behavior, sizeof(int) * N);

    printf("Running setup . . .\n");
	setup_kernel<<<80, 128>>>(d_state);
	cudaDeviceSynchronize();

	printf("Running popsicle_sticks_kernel . . .\n");
	popsicle_sticks_kernel<<<128, 128>>>(d_number_grid, d_popsicle_sticks, grid_dim, N, d_state, d_good_behavior);

	cudaDeviceSynchronize();

	// copy data to host for analysis.
	printf("Getting data back . . .\n");
	cudaMemcpy(good_behavior, d_good_behavior, N*sizeof(int), cudaMemcpyDeviceToHost);

	// calculate average
	int good_behavior_big;
	for (int i = 0; i < N; i++){
		good_behavior_big += good_behavior[i];
	}

	printf("%f\n", (float)good_behavior_big / (float)N);

	return 0;
}
