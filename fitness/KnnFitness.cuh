#pragma once

#include "../files/dataset.h"
#include "../genetics/Population.cuh"
#include <iomanip>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

struct Neabour {
    int label;
    int index;
};

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
__global__ void calculateNearestNeabours(float *distances, Neabour *neabours,
                                         LabelDataPoint<atributeCount> *points,
                                         const int size) {
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    const int idy = threadIdx.y + blockDim.y * blockIdx.y;
    if (idx >= size || idy >= size) {
        return;
    }
    distances[idy * size + idx] = distanceSquared(points[idy], points[idx]);
    neabours[idy * size + idx] = {points[idx].label, idx};

    // printf("idx: %i, idy: %i\n", idx, idy);
}

template <int atributeCount, int k>
__global__ void accuracyKernel(const DevicePopulation<bool> population,
                               const Neabour *nearestNeabours,
                               float *accuracy) {
    const int populationId = threadIdx.x + blockDim.x * blockIdx.x;
    const int rowId = threadIdx.y + blockDim.y * blockIdx.y;

    if (populationId >= population.popSize || rowId >= population.genSize) {
        return;
    }

    const auto specimen = getSpecimen(population, populationId);
    const auto neabours = nearestNeabours + rowId * population.genSize;

    int correctCount = 0;
    int foundCount = 0;

    for (size_t i = 1; i < population.genSize; i++) {
        if (specimen[neabours[i].index]) {
            foundCount++;
            correctCount += neabours[i].label == neabours[0].label ? 1 : 0;
            if (foundCount >= k) {
                if (correctCount >= (k / 2) + 1) {
                    atomicAdd(&accuracy[populationId], 1.f);
                }
                break;
            }
        }
    }
}

template <int atributeCount, int k> struct KnnFitness {

    thrust::device_vector<Neabour> nearestNeabours;
    thrust::device_vector<float> sums;

    int genSize() const { return atributeCount; }
    int dataSize;

    KnnFitness(const thrust::host_vector<LabelDataPoint<atributeCount>> &data)
        : nearestNeabours(data.size() * data.size()), dataSize(data.size()) {

        thrust::device_vector<float> distances(data.size() * data.size());

        const dim3 perBlock = {32, 32, 1};
        const dim3 blocks = {(unsigned int)data.size() / perBlock.x + 1,
                             (unsigned int)data.size() / perBlock.x + 1, 1};

        thrust::device_vector<LabelDataPoint<atributeCount>> deviceData = data;

        calculateNearestNeabours<<<blocks, perBlock>>>(
            thrust::raw_pointer_cast(&distances[0]),
            thrust::raw_pointer_cast(&nearestNeabours[0]),
            thrust::raw_pointer_cast(&deviceData[0]), (int)data.size());

        thrust::host_vector<cudaStream_t> streams(data.size());
        for (size_t i = 0; i < data.size(); i++) {
            cudaStreamCreate(&streams[i]);
            thrust::sort_by_key(thrust::cuda::par.on(streams[i]),
                                distances.begin() + i * data.size(),
                                distances.begin() + (i + 1) * data.size(),
                                nearestNeabours.begin() + i * data.size());
            cudaStreamDestroy(streams[i]);
        }

        for (size_t i = 0; i < data.size(); i++) {
            for (size_t j = 0; j < data.size(); j++) {
                std::cout
                    << std::setprecision(4)
                    << ((Neabour)nearestNeabours[i * data.size() + j]).label
                    << "|"
                    << ((Neabour)nearestNeabours[i * data.size() + j]).index
                    << "\t|";
            }
            std::cout << std::endl;
        }
    }

    void accuracy(Population<bool> &population,
                  thrust::device_vector<float> &accuracy) {
        thrust::fill(accuracy.begin(), accuracy.end(), 0.f);
        const dim3 threadsPerBlock = {32, 32, 1};
        const dim3 blocks = {(population.popSize() / threadsPerBlock.x) + 1,
                             (population.popSize() / threadsPerBlock.y) + 1, 1};

        accuracyKernel<atributeCount, k><<<blocks, threadsPerBlock>>>(
            population.toDevicePopulation(),
            thrust::raw_pointer_cast(&nearestNeabours[0]),
            thrust::raw_pointer_cast(&accuracy[0]));

        thrust::transform(accuracy.begin(), accuracy.end(), accuracy.begin(),
                          [dataSize = dataSize] __device__(float acc) {
                              return acc / dataSize;
                          });
    }

    auto &operator()(Population<bool> &population,
                     thrust::device_vector<float> &fitness) {

        return fitness;
    }
};
