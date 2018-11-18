#pragma once

#include "./Population.cuh"
#include <builtin_types.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>

template <typename T> struct RouletteSelection {

    float fitnessSum;

    __host__ void iteration(const Population<T> &population,
                            const thrust::device_vector<float> &fitness) {
        fitnessSum = thrust::reduce(fitness.begin(), fitness.end(), 0.f,
                                    thrust::plus<float>());
    }

    __device__ const T *operator()(const DevicePopulation<T> &population,
                                   const float *fitness) {
        const int idx = threadIdx.x + blockDim.x * blockIdx.x;

        curandState state;
        curand_init(clock64(), idx, 0, &state);
        const float randomNumber = curand_uniform(&state) * fitnessSum;
        float offset = 0.f;

        for (size_t i = 0;; i++) {
            offset += fitness[i];
            if (offset >= randomNumber) {
                return getSpecimen(population, i);
            }
        }
        // printf("should never hapen");
        // return nullptr;
    }
};
