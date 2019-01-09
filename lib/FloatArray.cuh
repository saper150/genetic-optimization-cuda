#pragma once
template <int count> struct FloatArray {
  float data[count];
  __device__ __host__ float operator[](const int index) { return data[index]; }
};