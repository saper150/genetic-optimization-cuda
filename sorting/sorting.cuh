#pragma once
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/partition.h>
#include <thrust/sequence.h>
#include "../DeviceArray.h"
#include "../lib/FloatArray.cuh"

struct SortingElement {
  int index;
  int dominatedByCount;
  bool* dominates;
};

inline __global__ void reduceDominatedByCount(int* indeces,
                                              DeviceArray<bool> dominates,
                                              DeviceArray<int> dominatesCount,
                                              int size) {
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;

  if (idx >= size || idy > dominatesCount.size) {
    return;
  }

  const bool d = dominates[indeces[idx] * dominatesCount.size + idy];

  if (d) {
    atomicAdd(dominatesCount.data + idy, -1);
  }
};

struct GetIndex : public thrust::unary_function<SortingElement, int> {
  __host__ __device__ int operator()(SortingElement el) const {
    return el.index;
  }
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

using DominanceGroups = thrust::host_vector<thrust::device_ptr<int>>;

template <int cryteriaCount>
__global__ void dominationKernel(
    DeviceArray<bool> dominates,
    DeviceArray<FloatArray<cryteriaCount>> fitnesses,
    DeviceArray<int> dominanceCounts) {
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;
  if (idx >= fitnesses.size || idy >= fitnesses.size || idx == idy) {
    return;
  }

  if (isDominating(fitnesses[idx], fitnesses[idy])) {
    dominates.data[fitnesses.size * idx + idy] = true;
  } else if (isDominating(fitnesses[idy], fitnesses[idx])) {
    atomicAdd(dominanceCounts.data + idx, 1);
  }
}

template <int cryteriaCount>
struct NonDominatedSorting {
  thrust::device_vector<int> indices;
  thrust::device_vector<int> dominanceCounts;
  thrust::device_vector<bool> dominates;
  int popSize;
  NonDominatedSorting(int popSize)
      : dominates(popSize * popSize, false),
        popSize(popSize),
        indices(popSize),
        dominanceCounts(popSize, 0) {}

  thrust::host_vector<thrust::device_ptr<int>> sort(
      thrust::device_vector<FloatArray<cryteriaCount>>& fitnesses) {
    InitializeSorting(fitnesses);

    thrust::host_vector<thrust::device_ptr<int>> groups;

    groups.push_back(&indices.begin()[0]);

    while (true) {
      const auto&& res = thrust::partition(
          groups.back(), &indices.end()[0],
          [dominanceCounts = thrust::raw_pointer_cast(
               &dominanceCounts[0])] __device__(const int el) {
            return dominanceCounts[el] == 0;
          });

      if (groups.back() - res == 0) {
        break;
      }
      groups.push_back(res);
      reduceDominanceCount(groups);
    }

    return groups;
  }

  thrust::host_vector<thrust::device_ptr<int>> sortHalf(
      thrust::device_vector<FloatArray<cryteriaCount>>& fitnesses) {
    InitializeSorting(fitnesses);

    thrust::host_vector<thrust::device_ptr<int>> groups;

    groups.push_back(&indices.begin()[0]);

    while (true) {
      const auto&& res = thrust::partition(
          groups.back(), &indices.end()[0],
          [dominanceCounts = thrust::raw_pointer_cast(
               &dominanceCounts[0])] __device__(const int el) {
            return dominanceCounts[el] == 0;
          });

      if (groups.back() - res == 0) {
        break;
      }
      groups.push_back(res);
      if (groups.back() - groups.front() >= popSize / 2) {
        break;
      }
      reduceDominanceCount(groups);
    }

    return groups;
  }

  void InitializeSorting(
      thrust::device_vector<FloatArray<cryteriaCount>>& fitnesses) {
    thrust::sequence(this->indices.begin(), this->indices.end());

    const dim3 perBlock = {32, 32, 1};
    const dim3 blocks = {(unsigned int)popSize / perBlock.x + 1,
                         (unsigned int)popSize / perBlock.y + 1, 1};
    dominationKernel<cryteriaCount><<<blocks, perBlock>>>(
        toDeviceArray(dominates), toDeviceArray(fitnesses),
        toDeviceArray(dominanceCounts));
  }

  thrust::device_ptr<int> getLastGroup(const DominanceGroups& groups) {
    return *(groups.begin() + groups.size() - 2);
  }

  void reduceDominanceCount(const DominanceGroups& groups) {
    auto lastGroup = getLastGroup(groups);
    const int size = groups.back() - lastGroup;

    const dim3 perBlock = {32, 32, 1};
    const dim3 blocks = {(unsigned int)size / perBlock.x + 1,
                         (unsigned int)popSize / perBlock.y + 1, 1};

    reduceDominatedByCount<<<blocks, perBlock>>>(
        thrust::raw_pointer_cast(&lastGroup[0]), toDeviceArray(dominates),
        toDeviceArray(dominanceCounts), size);
  }
};
