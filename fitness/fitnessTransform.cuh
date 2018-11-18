#pragma once

struct FitnessTransform {

    int dataSize;
    FitnessTransform(int dataSize);

    __device__ float operator()(const float f) const;
};
