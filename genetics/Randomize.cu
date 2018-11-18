#include "./Randomize.cuh"

#include <cmath>
#include <curand.h>
#include <curand_kernel.h>

__global__ void randomize(bool *f, const int size) {

    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx > size) {
        return;
    }
    curandState state;
    curand_init(clock64(), idx, 0, &state);
    f[idx] = curand_uniform(&state) > .5f ? true : false;
}

void randomize(thrust::device_vector<bool> &vec) {

    const int threadsPerBlock = 200;
    const int blocks = std::ceil((float)vec.size() / threadsPerBlock);
    randomize<<<blocks, threadsPerBlock>>>(thrust::raw_pointer_cast(&vec[0]),
                                           vec.size());
}
