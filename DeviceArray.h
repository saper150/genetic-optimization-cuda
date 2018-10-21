#pragma once

#include <thrust/device_vector.h>

template <typename T> struct DeviceArray {
    T *data;
    int size;
    __device__ T operator[](int i) { return data[i]; }
};

template <typename T>
DeviceArray<T> toDeviceArray(thrust::device_vector<T> &v) {
    return {thrust::raw_pointer_cast(&v[0]), static_cast<int>(v.size())};
};