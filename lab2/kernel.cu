#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <chrono>
#include <cuda_runtime.h>
#include <algorithm>


__global__ void bitonic_sort(int *gpuArr, int i, int j) {
    int k = threadIdx.x + blockDim.x * blockIdx.x;
    int xor_idx = k ^ (1 << j);
    if (xor_idx > k) {
        if ((1 << i) & k == 0) {
            if (gpuArr[k] > gpuArr[xor_idx]) {
                int temp = gpuArr[k];
                gpuArr[k] = gpuArr[xor_idx];
                gpuArr[xor_idx] = temp;
            }
        } else {
            if (gpuArr[k] < gpuArr[xor_idx]) {
                int temp = gpuArr[k];
                gpuArr[k] = gpuArr[xor_idx];
                gpuArr[xor_idx] = temp;                
            }
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

    for (int i = 1; i <= log2(size); i++) {
        for (int j = i - 1; j >= 0; j++) {
            bitonic_sort<<<size / 1024, 1024>>>(gpuArr, i, j);
            cudaMemcpy(arrSortedGpu, gpuArr, size * sizeof(int), cudaMemcpyDeviceToHost);
            for (int i = 0; i < size; i++) {
                printf("arr[%d] = %d\n", i, arrSortedGpu[i]);
            }
        }
    }

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

    for (int i = 0; i < size; i++) {
        printf("arr[%d] = %d\n", i, arrSortedGpu[i]);
    }
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

