#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <iomanip>
#include <fstream>
#include <limits>

#define CUDA_CORES 768  // GTX 1050 Ti CUDA cores
#define BLOCK_SIZE 256  // Best for 1050 Ti
#define WARP_SIZE 32

#define CUDA_CHECK(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    }

// Warp reduction function
__inline__ __device__ int warpReduceSum(int val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

// Hierarchical block-level reduction using shared memory
__global__ void reduceSum(int* input, int* output, int n) {
    __shared__ int sharedData[BLOCK_SIZE]; 

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int localSum = (tid < n) ? input[tid] : 0;
    
    // Perform warp-level reduction
    localSum = warpReduceSum(localSum);

    // Store warp results in shared memory
    int lane = threadIdx.x % WARP_SIZE;
    int warpId = threadIdx.x / WARP_SIZE;

    if (lane == 0) sharedData[warpId] = localSum;
    __syncthreads();

    // Reduce shared memory
    if (warpId == 0) {
        localSum = (lane < (blockDim.x / WARP_SIZE)) ? sharedData[lane] : 0;
        localSum = warpReduceSum(localSum);
    }

    // First thread writes block result
    if (threadIdx.x == 0) atomicAdd(output, localSum);
}

// Sequential CPU Sum
long long sequentialSum(const std::vector<int>& data) {
    long long sum = 0;
    for (int val : data) sum += val;
    return sum;
}

// Sequential CPU Min
int sequentialMin(const std::vector<int>& data) {
    int minVal = std::numeric_limits<int>::max();
    for (int val : data) minVal = std::min(minVal, val);
    return minVal;
}

// Sequential CPU Average
double sequentialAverage(const std::vector<int>& data) {
    return static_cast<double>(sequentialSum(data)) / data.size();
}

int main() {
    std::vector<long long> sizes = {95000, 520000, 930000, 1110000,4600000 };
    std::vector<int> maxValues = {1200, 2400, 4600, 5800};
    std::cout << "\n";
 std::cout << "\n";
 std::cout << "Name:ABC no:123 CLASS: BE \n";
 std::cout << "\n";
 std::cout << "\n";
    std::cout << "---------------------------------------------------------------------------------------------------\n";
    std::cout << "| Input Size | Max Value | CPU Sum       | GPU Sum       | CPU Time (s) | GPU Time (s) | Speedup  | Efficiency | Avg Value | Min Value |\n";
    std::cout << "---------------------------------------------------------------------------------------------------\n";

    for (size_t i = 0; i < sizes.size(); i++) {
        long long n = sizes[i];
        int maxValue = maxValues[i];

        std::vector<int> data(n);
        for (long long j = 0; j < n; ++j) {
            data[j] = rand() % maxValue;
        }

        int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

        int* d_input;
        int* d_output;
        int h_output = 0;

        CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_output, sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_input, data.data(), n * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_output, 0, sizeof(int)));

        // CPU Calculation
        auto startSeq = std::chrono::high_resolution_clock::now();
        long long seqSum = sequentialSum(data);
        int seqMin = sequentialMin(data);
        double seqAvg = sequentialAverage(data);
        auto endSeq = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> seqTime = endSeq - startSeq;

        // GPU Calculation
        float gpuSumTime;
        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);

        cudaEventRecord(start);
        reduceSum<<<numBlocks, BLOCK_SIZE>>>(d_input, d_output, n);
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&gpuSumTime, start, end);

        CUDA_CHECK(cudaMemcpy(&h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost));

        // Speedup Calculation
        double speedup = seqTime.count() / (gpuSumTime / 1000.0);

        // Efficiency Calculation
        double efficiency = speedup / CUDA_CORES;

        // Print Results in Tabular Format
        std::cout << "| " << std::setw(10) << n
                  << " | " << std::setw(9) << maxValue
                  << " | " << std::setw(12) << seqSum
                  << " | " << std::setw(12) << h_output
                  << " | " << std::setw(11) << std::fixed << std::setprecision(6) << seqTime.count()
                  << " | " << std::setw(11) << gpuSumTime / 1000.0
                  << " | " << std::setw(7) << std::fixed << std::setprecision(2) << speedup
                  << " | " << std::setw(10) << std::fixed << std::setprecision(6) << efficiency
                  << " | " << std::setw(9) << std::fixed << std::setprecision(2) << seqAvg
                  << " | " << std::setw(9) << seqMin
                  << " |\n";

        cudaFree(d_input);
        cudaFree(d_output);
    }

    std::cout << "---------------------------------------------------------------------------------------------------\n";
    return 0;
}

