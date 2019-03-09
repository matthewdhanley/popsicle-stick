#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include "gpu_err_check.h"


// Print device properties
void printDevProp(cudaDeviceProp devProp)
{
    printf("Name:                          %s\n\n",  devProp.name);
    printf("Major revision number:         %d\n",  devProp.major);
    printf("Minor revision number:         %d\n",  devProp.minor);
    printf("Clock rate:                    %d kHz\n\n",  devProp.clockRate);

    printf("Total global memory:           %lu bytes\n",  devProp.totalGlobalMem);
    printf("Total shared memory per block: %lu bytes\n",  devProp.sharedMemPerBlock);
    printf("Total shared memory per SM:    %lu bytes\n",  devProp.sharedMemPerMultiprocessor);
    printf("Total registers per block:     %d\n",  devProp.regsPerBlock);
    printf("Total registers per SM:        %d\n",  devProp.regsPerMultiprocessor);
    printf("Maximum memory pitch:          %lu bytes\n",  devProp.memPitch);
    printf("\t(Maximum pitch in bytes\nallowed by memory copies)\n");
    printf("Total constant memory:         %lu bytes\n",  devProp.totalConstMem);
    printf("L2 Cache Size:                 %d bytes\n",  devProp.l2CacheSize);
    printf("Supports caching locals in L1: %s\n",  (devProp.localL1CacheSupported ? "Yes" : "No"));
    printf("Supports caching globals in L1: %s\n",  (devProp.globalL1CacheSupported ? "Yes" : "No"));
    printf("Memory Bus Width:              %d bits\n",  devProp.memoryBusWidth);
    printf("Memory Clock Rate:             %d kHz\n\n",  devProp.memoryClockRate);

    printf("Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
    printf("Warp size:                     %d threads\n",  devProp.warpSize);
    printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
    for (int i = 0; i < 3; ++i) {
        printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
    }
    for (int i = 0; i < 3; ++i) {
        printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
    }
    printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
    printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
    return;
}




void get_device_info(int dev_number){
    cudaDeviceProp d_info;
    cudaSafeCall( cudaGetDeviceProperties(&d_info, dev_number) );
    printDevProp(d_info);
}


int main(){
    // Get device info
    get_device_info(0);
    return 0;
}

