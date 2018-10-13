#pragma once

#include "../genetics/Population.cuh"
#include <stdio.h>
#include <thrust/device_vector.h>

template <typename T>
__global__ void fitnessKernel(float *fitness, const DevicePopulation<T> p) {
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    const auto gen = getSpecimen(p, idx);
    fitness[idx] = 0.f;
    for (int i = 0; i < p.genSize; i++) {
        if ((i % 2 == 0) && gen[i]) {
            fitness[idx] += 1.f;
        }
        if ((i % 2 != 0) && !gen[i]) {
            fitness[idx] += 1.f;
        }
    }
    fitness[idx] = fitness[idx] * fitness[idx] * fitness[idx];
}

template <typename T> struct BasicFitness {
  private:
    int m_genSize;

  public:
    int genSize() const { return m_genSize; }

    BasicFitness(int genSize) : m_genSize(genSize) {}

    void operator()(Population<T> &population,
                    thrust::device_vector<float> &fitness) {

        fitnessKernel<T>
            <<<1, population.popSize()>>>(thrust::raw_pointer_cast(&fitness[0]),
                                          population.toDevicePopulation());

        const auto max = thrust::max_element(fitness.begin(), fitness.end());
        std::cout << *max << std::endl;
    }
};
