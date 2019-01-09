#pragma once

template <typename T> struct Mutation {
    float mutationRate;
    __device__ void operator()(T *gen, const int genSize) {
        const int idx = threadIdx.x + blockDim.x * blockIdx.x;
        curandState state;
        curand_init(clock64(), idx, 0, &state);
        const float randomNumber = curand_uniform(&state);

        for (int i = 0; i < genSize; i++) {
            if (curand_uniform(&state) <= mutationRate) {
                gen[i] = !gen[i];
            }
        }
    }
};
