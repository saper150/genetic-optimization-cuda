#pragma once
#include "../genetics/Population.cuh"

inline __global__ void crossoverKernel(const bool* src1,
                                       const bool* src2,
                                       bool* desc,
                                       const int* rng,
                                       const thrust::tuple<int, int>* pairs,
                                       const int popSize,
                                       const int genSize) {
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;

  if (idx >= popSize || idy >= genSize) {
    return;
  }

  const bool* p1 = pairs[idx].get<0>() >= popSize
                       ? src2 + ((pairs[idx].get<0>() - popSize) * genSize)
                       : src1 + (pairs[idx].get<0>() * genSize);
  
  const bool* p2 = pairs[idx].get<1>() >= popSize
                       ? src2 + ((pairs[idx].get<1>() - popSize) * genSize)
                       : src1 + (pairs[idx].get<1>() * genSize);

  bool* currentDestination = desc + (idx * genSize);
  currentDestination[idy] = idy < rng[idx] ? p1[idy] : p2[idy];
}

template <typename T>
struct Crossover {
  void cross(const Population<T>& src1,
             const Population<T>& src2,
             Population<T>& dest,
             const thrust::device_vector<thrust::tuple<int, int>>& pairs,
             const thrust::device_vector<int>& rng) {
    const dim3 threadsPerBlock = {32, 32, 1};
    const dim3 blocks = {(src1.popSize() / threadsPerBlock.x) + 1,
                         (src1.popSize() / threadsPerBlock.y) + 1, 1};
    crossoverKernel<<<blocks, threadsPerBlock>>>(
        thrust::raw_pointer_cast(src1.population.data()),
        thrust::raw_pointer_cast(src2.population.data()),
        thrust::raw_pointer_cast(dest.population.data()),
        thrust::raw_pointer_cast(rng.data()),
        thrust::raw_pointer_cast(pairs.data()), src1.popSize(), src1.genSize);
  }
};
