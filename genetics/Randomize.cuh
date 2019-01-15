#pragma once

#include <thrust/device_vector.h>
#include <curand.h>

void randomize(thrust::device_vector<bool> &vec);

void randomize(thrust::device_vector<int> &vec, int max);

struct IntRng {
    int max;
    curandGenerator_t generator;
    thrust::device_vector<float> intermidiate;
    thrust::device_vector<int> res;
    thrust::device_vector<int>& generate();
    IntRng(int size, int max);
    ~IntRng();
};