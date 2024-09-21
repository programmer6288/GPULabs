#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <chrono>
#include <cuda_runtime.h>
#include <algorithm>

#define SHARED_MEM_SIZE 1024

// CUDA kernel to perform bitonic sort in shared memory for smaller subsequences
__global__ void bitonicSortShared(int *data, int numElements) {
    extern __shared__ int sharedData[];

    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int tid = threadIdx.x;

    // Load data from global memory to shared memory
    if (idx < numElements) {
        sharedData[tid] = data[idx];
    }
    __syncthreads();

    // Perform bitonic sort in shared memory
    for (int k = 2; k <= blockDim.x; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            unsigned int ixj = tid ^ j;
            if (ixj > tid) {
                if ((tid & k) == 0) {
                    // Ascending order
                    if (sharedData[tid] > sharedData[ixj]) {
                        int temp = sharedData[tid];
                        sharedData[tid] = sharedData[ixj];
                        sharedData[ixj] = temp;
                    }
                } else {
                    // Descending order
                    if (sharedData[tid] < sharedData[ixj]) {
                        int temp = sharedData[tid];
                        sharedData[tid] = sharedData[ixj];
                        sharedData[ixj] = temp;
                    }
                }
            }
            __syncthreads();
        }
    }

    // Write the sorted subsequence back to global memory
    if (idx < numElements) {
        data[idx] = sharedData[tid];
    }
}
// CUDA kernel to perform global memory bitonic merge (subsequences already sorted)
__global__ void bitonicMergeGlobal(int *data, int sortedSubseqSize, int numElements) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Start merging from sorted subsequence size
    for (int k = sortedSubseqSize * 2; k <= numElements; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            unsigned int ixj = idx ^ j;
            if (ixj > idx && ixj < numElements) {
                if ((idx & k) == 0) {
                    // Ascending order
                    if (data[idx] > data[ixj]) {
                        int temp = data[idx];
                        data[idx] = data[ixj];
                        data[ixj] = temp;
                    }
                } else {
                    // Descending order
                    if (data[idx] < data[ixj]) {
                        int temp = data[idx];
                        data[idx] = data[ixj];
                        data[ixj] = temp;
                    }
                }
            }
            __syncthreads();
        }
    }
}


int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <array_size>\n", argv[0]);
        return 1;
    }

    int size = atoi(argv[1]);

    srand(time(NULL));

    // ======================================================================
    // arCpu contains the input random array
    // arrSortedGpu should contain the sorted array copied from GPU to CPU
    // ======================================================================
    int* arrCpu = (int*)malloc(size * sizeof(int));
    int* arrSortedGpu = (int*)malloc(size * sizeof(int));

    for (int i = 0; i < size; i++) {
        arrCpu[i] = rand() % 1000;
    }

    float gpuTime, h2dTime, d2hTime, cpuTime = 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    // ======================================================================
    // Transfer data (arr_cpu) to device
    // ======================================================================

    // your code goes here .......
    // int modSize = 1 << ceil(log2(size));
    int *gpuArr;
    cudaMalloc(&gpuArr, size * sizeof(int));
    cudaMemcpy(gpuArr, arrCpu, size * sizeof(int), cudaMemcpyHostToDevice);
    

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&h2dTime, start, stop);

    cudaEventRecord(start);
    
    // ======================================================================
    // Perform bitonic sort on GPU
    // ======================================================================

    // your code goes here .......

    // Launch the shared memory bitonic sort for smaller subsequences
    dim3 blocks(size / SHARED_MEM_SIZE);
    dim3 threads(SHARED_MEM_SIZE);
    bitonicSortShared<<<blocks, threads, SHARED_MEM_SIZE * sizeof(int)>>>(gpuArr, size);
    cudaDeviceSynchronize();


    // Launch the global memory kernel to merge the results (only once)
    dim3 blocksGlobal(size / 1024);
    dim3 threadsGlobal(1024);
    
    // The sortedSubseqSize is the size of subsequences already sorted by shared memory (e.g., 1024)
    bitonicMergeGlobal<<<blocksGlobal, threadsGlobal>>>(gpuArr, SHARED_MEM_SIZE, size);
    cudaDeviceSynchronize();


    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);

    cudaEventRecord(start);

    // ======================================================================
    // Transfer sorted data back to host (copied to arr_sorted_gpu)
    // ======================================================================

    // your code goes here .......
    cudaMemcpy(arrSortedGpu, gpuArr, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(gpuArr);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&d2hTime, start, stop);

    auto startTime = std::chrono::high_resolution_clock::now();
    
    // CPU sort for performance comparison
    std::sort(arrCpu, arrCpu + size);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    cpuTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
    cpuTime = cpuTime / 1000;

    int match = 1;
    for (int i = 0; i < size; i++) {
        if (arrSortedGpu[i] != arrCpu[i]) {
            match = 0;
            break;
        }
    }

    free(arrCpu);
    free(arrSortedGpu);

    if (match)
        printf("\033[1;32mFUNCTIONAL SUCCESS\n\033[0m");
    else {
        printf("\033[1;31mFUNCTIONCAL FAIL\n\033[0m");
        return 0;
    }
    
    printf("\033[1;34mArray size         :\033[0m %d\n", size);
    printf("\033[1;34mCPU Sort Time (ms) :\033[0m %f\n", cpuTime);
    float gpuTotalTime = h2dTime + gpuTime + d2hTime;
    int speedup = (gpuTotalTime > cpuTime) ? (gpuTotalTime/cpuTime) : (cpuTime/gpuTotalTime);
    float meps = size / (gpuTotalTime * 0.001) / 1e6;
    printf("\033[1;34mGPU Sort Time (ms) :\033[0m %f\n", gpuTotalTime);
    printf("\033[1;34mGPU Sort Speed     :\033[0m %f million elements per second\n", meps);
    if (gpuTotalTime < cpuTime) {
        printf("\033[1;32mPERF PASSING\n\033[0m");
        printf("\033[1;34mGPU Sort is \033[1;32m %dx \033[1;34mfaster than CPU !!!\033[0m\n", speedup);
        printf("\033[1;34mH2D Transfer Time (ms):\033[0m %f\n", h2dTime);
        printf("\033[1;34mKernel Time (ms)      :\033[0m %f\n", gpuTime);
        printf("\033[1;34mD2H Transfer Time (ms):\033[0m %f\n", d2hTime);
    } else {
        printf("\033[1;31mPERF FAILING\n\033[0m");
        printf("\033[1;34mGPU Sort is \033[1;31m%dx \033[1;34mslower than CPU, optimize further!\n", speedup);
        printf("\033[1;34mH2D Transfer Time (ms):\033[0m %f\n", h2dTime);
        printf("\033[1;34mKernel Time (ms)      :\033[0m %f\n", gpuTime);
        printf("\033[1;34mD2H Transfer Time (ms):\033[0m %f\n", d2hTime);
        return 0;
    }

    return 0;
}

