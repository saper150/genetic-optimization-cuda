#include "./fitnessTransform.cuh"

FitnessTransform::FitnessTransform(int dataSize) : dataSize(dataSize) {}

__device__ float FitnessTransform::operator() (const float f) const {
    return f / dataSize;
}
