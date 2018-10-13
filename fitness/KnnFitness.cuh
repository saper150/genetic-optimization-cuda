#pragma once

#include "../files/dataset.h"
#include "../genetics/Population.cuh"
#include <thrust/host_vector.h>
#include <iomanip>

template <int atributeCount>
__device__ float distanceSquared(LabelDataPoint<atributeCount> a,
                                 LabelDataPoint<atributeCount> b) {

    float acc = 0.f;

    for (size_t i = 0; i < atributeCount; i++) {
        acc +=
            (a.position[i] - b.position[i]) * (a.position[i] - b.position[i]);
    }
    return acc;
}

template <int atributeCount>
__global__ void calculateNearestNeabours(float *distances,
                                         LabelDataPoint<atributeCount> *points,
                                         const int size) {
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    const int idy = threadIdx.y + blockDim.y * blockIdx.y;
    if (idx > size || idy > size) {
        return;
    }
    distances[idy * size + idx] = distanceSquared(points[idy], points[idx]);
    printf("idx: %i, idy: %i\n", idx, idy);
}

template <int atributeCount> struct KnnFitness {

    thrust::device_vector<int> nearestNeabours;

    int genSize() const { return atributeCount; }

    KnnFitness(const thrust::host_vector<LabelDataPoint<atributeCount>> &data)
        : nearestNeabours(data.size() * data.size()) {

        thrust::device_vector<float> distances(data.size() * data.size());

        const dim3 perBlock = {32, 32, 1};
        const dim3 blocks = {(unsigned int)data.size() / perBlock.x + 1,
                             (unsigned int)data.size() / perBlock.x + 1, 1};

        thrust::device_vector<LabelDataPoint<atributeCount>> deviceData = data;
        calculateNearestNeabours<<<blocks, perBlock>>>(
            thrust::raw_pointer_cast(&distances[0]),
            thrust::raw_pointer_cast(&deviceData[0]), (int)data.size());
        std::cout << distances[4] << " wut ---------------------" << std::endl;
        // std::cout.precision(5);
        for (size_t i = 0; i < data.size(); i++) {
            for (size_t j = 0; j < data.size(); j++) {
                std::cout << std::setprecision(4)
                          << distances[i * data.size() + j] << "\t|";
            }
            std::cout << std::endl;
        }
    }

    auto &operator()(Population<bool> &population,
                     thrust::device_vector<float> &fitness) {
        return fitness;
    }
};

// template <typename T> struct Fitness {
//   private:
//     thrust::device_vector<float> fitness;
//     int popSize;

//   public:
//     Fitness(int popSize, int genSize)
//         : fitness(popSize, 0.0f), popSize(popSize) {}

//     const thrust::device_vector<float> &operator()(Population<T> &population)
//     {

//         fitnessKernel<T><<<1,
//         popSize>>>(thrust::raw_pointer_cast(&fitness[0]),
//                                          population.toDevicePopulation());
//         return fitness;
//     }
// };

// template <typename T>
// __global__ void fitnessKernel(float *fitness, const DevicePopulation<T> p) {
//     const int idx = threadIdx.x + blockDim.x * blockIdx.x;
//     const auto gen = getSpecimen(p, idx);
//     fitness[idx] = 0.f;
//     for (int i = 0; i < p.genSize; i++) {
//         // fitness[idx] += gen[i];
//         if ((i % 2 == 0) && gen[i]) {
//             fitness[idx] += 1.f;
//         }
//         if ((i % 2 != 0) && !gen[i]) {
//             fitness[idx] += 1.f;
//         }
//     }
//     fitness[idx] *= fitness[idx];
// }