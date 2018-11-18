#pragma once
#include "../uniform_int.h"

template <typename T> struct SinglePointCrossover {
    __device__ void operator()(const int genSize, T *copyInto, const T *gen1,
                               const T *gen2) {
        const int idx = threadIdx.x + blockDim.x * blockIdx.x;
        curandState state;
        curand_init(clock64(), idx, 0, &state);
        const int crossOverPoint = uniform_int(state, genSize);
        for (size_t i = 0; i < crossOverPoint; i++) {
            copyInto[i] = gen1[i];
        }

        for (size_t i = crossOverPoint; i < genSize; i++) {
            copyInto[i] = gen2[i];
        }
    }
};

template <typename T, int pointCount> struct MultiPointCrossover {
    __device__ void operator()(const int genSize, T *copyInto, const T *gen1,
                               const T *gen2) {
        const int idx = threadIdx.x + blockDim.x * blockIdx.x;
        curandState state;
        curand_init(clock64(), idx, 0, &state);
        int points[pointCount];

        for (size_t i = 0; i < pointCount; i++) {
            points[i] = uniform_int(state, genSize);
        }

        for (size_t i = 0; i < points[0]; i++) {
            copyInto[i] = gen1[i];
        }

        const T *gens[] = {gen1, gen2};
        for (size_t i = 0; i < pointCount - 1; i++) {
            const auto copyFrom = gens[(i + 1) % 2];
            for (size_t j = points[i]; j < points[i + 1]; j++) {
                copyInto[j] = copyFrom[j];
            }
        }

        for (size_t i = points[pointCount - 1]; i < genSize; i++) {
            copyInto[i] = gen2[i];
        }
    }
};
