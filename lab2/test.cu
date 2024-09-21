// High-Performance Bitonic Sort with Shared Memory in CUDA
// Written in C++ with CUDA extensions

#include <cuda_runtime.h>
#include <stdio.h>

#define THREADS_PER_BLOCK 1024  // Must be a power of two
#define SHARED_SIZE_LIMIT 1024  // Adjust based on shared memory size per block

__global__ void bitonicSortSharedKernel(float *d_data, int size, int chunkSize)
{
    extern __shared__ float s_data[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * chunkSize + tid;

    // Load data into shared memory
    if (idx < size)
        s_data[tid] = d_data[idx];
    else
        s_data[tid] = FLT_MAX;  // Use a sentinel value for out-of-bounds indices

    __syncthreads();

    // Perform bitonic sort on shared memory
    for (int k = 2; k <= chunkSize; k <<= 1)
    {
        for (int j = k >> 1; j > 0; j >>= 1)
        {
            unsigned int ixj = tid ^ j;
            if (ixj < chunkSize)
            {
                if ((tid & k) == 0)
                {
                    if (s_data[tid] > s_data[ixj])
                    {
                        // Swap
                        float temp = s_data[tid];
                        s_data[tid] = s_data[ixj];
                        s_data[ixj] = temp;
                    }
                }
                else
                {
                    if (s_data[tid] < s_data[ixj])
                    {
                        // Swap
                        float temp = s_data[tid];
                        s_data[tid] = s_data[ixj];
                        s_data[ixj] = temp;
                    }
                }
            }
            __syncthreads();
        }
    }

    // Write sorted data back to global memory
    if (idx < size)
        d_data[idx] = s_data[tid];
}

__global__ void bitonicSortGlobalKernel(float *d_data, int size, int j, int k)
{
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= size)
        return;

    unsigned int ixj = idx ^ j;

    if (ixj > idx)
    {
        if ((idx & k) == 0)
        {
            if (d_data[idx] > d_data[ixj])
            {
                // Swap
                float temp = d_data[idx];
                d_data[idx] = d_data[ixj];
                d_data[ixj] = temp;
            }
        }
        else
        {
            if (d_data[idx] < d_data[ixj])
            {
                // Swap
                float temp = d_data[idx];
                d_data[idx] = d_data[ixj];
                d_data[ixj] = temp;
            }
        }
    }
}

void bitonicSort(float *d_data, int size)
{
    int numSubSequences = (size + SHARED_SIZE_LIMIT - 1) / SHARED_SIZE_LIMIT;

    // First, sort small subsequences using shared memory
    for (int i = 0; i < numSubSequences; ++i)
    {
        int offset = i * SHARED_SIZE_LIMIT;
        int currentChunkSize = min(SHARED_SIZE_LIMIT, size - offset);

        dim3 blocks(1);
        dim3 threads(currentChunkSize);

        size_t sharedMemSize = currentChunkSize * sizeof(float);

        bitonicSortSharedKernel<<<blocks, threads, sharedMemSize>>>(d_data + offset, size, currentChunkSize);

        cudaDeviceSynchronize();
    }

    // Now, perform the merging steps using global memory
    for (int k = SHARED_SIZE_LIMIT * 2; k <= size; k <<= 1)
    {
        for (int j = k >> 1; j > 0; j >>= 1)
        {
            dim3 blocks((size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
            dim3 threads(THREADS_PER_BLOCK);

            bitonicSortGlobalKernel<<<blocks, threads>>>(d_data, size, j, k);

            cudaDeviceSynchronize();
        }
    }
}

int main()
{
    // Example usage
    int size = 1 << 20;  // Must be a power of two
    size_t bytes = size * sizeof(float);

    float *h_data = (float *)malloc(bytes);

    // Initialize data
    for (int i = 0; i < size; i++)
    {
        h_data[i] = (float)(rand() % 1000);
    }

    float *d_data;
    cudaMalloc((void **)&d_data, bytes);
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);

    bitonicSort(d_data, size);

    cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost);

    // Verify the result
    bool sorted = true;
    for (int i = 0; i < size - 1; i++)
    {
        if (h_data[i] > h_data[i + 1])
        {
            sorted = false;
            break;
        }
    }

    if (sorted)
    {
        printf("Array is sorted.\n");
    }
    else
    {
        printf("Array is NOT sorted.\n");
    }

    cudaFree(d_data);
    free(h_data);

    return 0;
}
