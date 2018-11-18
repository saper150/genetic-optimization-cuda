#pragma once
#include "../DeviceArray.h"
#include <thrust/device_vector.h>

template <int count> struct FloatArray {
    float data[count];
    __device__ __host__ float operator[](const int index) {
        return data[index];
    }
};

struct SortingElement {
    int dominatedByCount;
    bool *dominates;
};

template <int cryteriaCount>
__device__ __host__ bool isDominating(FloatArray<cryteriaCount> a,
                                      FloatArray<cryteriaCount> b) {
    for (size_t i = 0; i < cryteriaCount; i++) {
        if (a[i] <= b[i]) {
            return false;
        }
    }
    return true;
}

template <int cryteriaCount>
__global__ void
dominationKernel(DeviceArray<SortingElement> elements,
                 DeviceArray<FloatArray<cryteriaCount>> fitnesses) {
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    const int idy = threadIdx.y + blockDim.y * blockIdx.y;
    if (idx >= elements.size || idy >= elements.size || idx == idy) {
        return;
    }
    if (isDominating(fitnesses[idx], fitnesses[idy])) {
        elements[idx].dominates[idy] = true;
    } else {
        int *a = &elements.data[idx].
        
        dominatedByCount;
        atomicAdd(a, 1);
    }
}

template <int cryteriaCount> struct InitializeSorting {
    bool *dominates;
    int popSize;
    InitializeSorting(bool *dominates, int popSize)
        : popSize(popSize), dominates(dominates){};
    __device__ SortingElement operator()(int i) {
        SortingElement c;
        c.dominates = dominates + i * popSize;
        return c;
    }
};

template <int cryteriaCount> struct NonDominatedSorting {

    thrust::device_vector<SortingElement> sortingElements;
    thrust::device_vector<bool> dominates;
    int popSize;
    NonDominatedSorting(int popSize)
        : sortingElements(popSize), dominates(popSize * popSize),
          popSize(popSize) {

        // const bool *deviceDominates =
        // thrust::raw_pointer_cast(&dominates[0]); const auto lambda =
        // [deviceDominates = deviceDominates,
        //                      popSize = popSize] __device__(int i) {
        //     SortingElement<cryteriaCount> c;
        //     c.dominatesBegin = deviceDominates + i * popSize;
        //     return c;
        // };

        thrust::transform(
            thrust::counting_iterator<int>(0),
            thrust::counting_iterator<int>(popSize), sortingElements.begin(),
            InitializeSorting<cryteriaCount>(
                thrust::raw_pointer_cast(&dominates[0]), popSize));
    }

    // thrust::device_vector<int> counters(0);
    void sort(thrust::device_vector<FloatArray<cryteriaCount>> &fitnesses,
              thrust::device_vector<int> &sorted) {
        // thrust::transform(sortingElements.begin(), sortingElements.end(),
        //                   sortingElements.begin(),
        //                   [] __device__(SortingElement el) {
        //                       el.dominatedByCount = 0;
        //                       return el;
        //                   });
        const dim3 perBlock = {32, 32, 1};
        const dim3 blocks = {(unsigned int)popSize / perBlock.x + 1,
                             (unsigned int)popSize / perBlock.y + 1, 1};
        dominationKernel<cryteriaCount><<<blocks, perBlock>>>(
            toDeviceArray(sortingElements), toDeviceArray(fitnesses));

        for (auto el : sortingElements) {
            const SortingElement s = el;
            std::cout << s.dominatedByCount << std::endl;
        }

        // sets.resize(fitnesses.size() * fitnesses.size());
        // counters.resize(fitnesses.size());
    }
};
