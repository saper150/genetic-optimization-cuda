#pragma once

#include "../DeviceArray.h"
#include "../genetics/Population.cuh"
#include "../files/dataset.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/transform_iterator.h>

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
__global__ void
nearestNeaboursKernel(float *distances, Neabour *neabours,
                      DeviceArray<LabelDataPoint<atributeCount>> teaching,
                      DeviceArray<LabelDataPoint<atributeCount>> test) {
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    const int idy = threadIdx.y + blockDim.y * blockIdx.y;
    if (idx >= teaching.size || idy >= test.size) {
        return;
    }
    distances[idy * teaching.size + idx] =
        distanceSquared(teaching[idx], test[idy]);
    neabours[idy * teaching.size + idx] = {teaching[idx].label, idx};
}

template <int atributeCount, int labelsCount, int k>
__global__ void sameAccuracyKernel(const DevicePopulation<bool> population,
                                   const Neabour *nearestNeabours,
                                   float *accuracy) {
    const int populationId = threadIdx.x + blockDim.x * blockIdx.x;
    const int rowId = threadIdx.y + blockDim.y * blockIdx.y;

    if (populationId >= population.popSize || rowId >= population.genSize) {
        return;
    }

    const auto specimen = getSpecimen(population, populationId);
    const auto neabours = nearestNeabours + rowId * population.genSize;

    int iterator = 0;
    int labelsCounts[labelsCount] = {0};

    for (size_t i = 1; i < population.genSize; i++) {
        if (specimen[neabours[i].index]) {
            labelsCounts[neabours[i].label]++;
            iterator++;
            if (iterator >= k) {
                break;
            }
        }
    }

    const auto iter = thrust::max_element(thrust::seq, labelsCounts,
                                          labelsCounts + labelsCount);
    const int position = iter - labelsCounts;

    if (position == neabours[0].label) {
        atomicAdd(&accuracy[populationId], 1.f);
    }
}

template <int atributeCount, int labelsCount, int k>
__global__ void accuracyKernel(const DevicePopulation<bool> population,
                               const Neabour *nearestNeabours, int *labels,
                               float *accuracy, int size) {
    const int populationId = threadIdx.x + blockDim.x * blockIdx.x;
    const int rowId = threadIdx.y + blockDim.y * blockIdx.y;

    if (populationId >= population.popSize || rowId >= size) {
        return;
    }
    const auto specimen = getSpecimen(population, populationId);
    const auto neabours = nearestNeabours + rowId * population.genSize;

    int foundCount = 0;
    int labelsCounts[labelsCount] = {0};

    for (size_t i = 0; i < population.genSize; i++) {
        if (specimen[neabours[i].index]) {
            labelsCounts[neabours[i].label]++;
            foundCount++;
            if (foundCount >= k) {
                break;
            }
        }
    }

    const auto iter = thrust::max_element(thrust::seq, labelsCounts,
                                          labelsCounts + labelsCount);
    const int position = iter - labelsCounts;
    if (position == labels[rowId]) {
        atomicAdd(&accuracy[populationId], 1.f);
    }
};

template <typename T> struct GetLabel : public thrust::unary_function<T, int> {
    __host__ __device__ float operator()(T a) const { return a.label; }
};

template <int atributeCount, int labelsCount, int k> class Knn {
  private:
    thrust::device_vector<Neabour> nearestNeabours;
    thrust::device_vector<int> labels;

    void
    fillLabels(const thrust::host_vector<LabelDataPoint<atributeCount>> &data) {
        labels.resize(data.size());
        const auto beginIter = thrust::make_transform_iterator(
            data.begin(), GetLabel<LabelDataPoint<atributeCount>>());
        const auto endIter = thrust::make_transform_iterator(
            data.end(), GetLabel<LabelDataPoint<atributeCount>>());

        thrust::copy(beginIter, endIter, labels.begin());

    }

    void calcNearestNeabours(
        thrust::device_vector<LabelDataPoint<atributeCount>> &traingData,
        thrust::device_vector<LabelDataPoint<atributeCount>> &testData) {
        nearestNeabours =
            thrust::device_vector<Neabour>(traingData.size() * testData.size());
        const dim3 perBlock = {32, 32, 1};
        const dim3 blocks = {(unsigned int)traingData.size() / perBlock.x + 1,
                             (unsigned int)testData.size() / perBlock.y + 1, 1};

        thrust::device_vector<float> distances(traingData.size() *
                                               testData.size());

        nearestNeaboursKernel<atributeCount><<<blocks, perBlock>>>(
            thrust::raw_pointer_cast(&distances[0]),
            thrust::raw_pointer_cast(&nearestNeabours[0]),
            toDeviceArray<LabelDataPoint<atributeCount>>(traingData),
            toDeviceArray<LabelDataPoint<atributeCount>>(testData));
        thrust::host_vector<cudaStream_t> streams(testData.size());
        for (size_t i = 0; i < testData.size(); i++) {
            cudaStreamCreate(&streams[i]);
            thrust::sort_by_key(thrust::cuda::par.on(streams[i]),
                                distances.begin() + i * traingData.size(),
                                distances.begin() + (i + 1) * traingData.size(),
                                nearestNeabours.begin() +
                                    i * traingData.size());
            cudaStreamDestroy(streams[i]);
        }
        // for (size_t i = 0; i < testData.size(); i++) {
        //     for (size_t j = 0; j < testData.size(); j++) {
        //         std::cout
        //             << ((Neabour)nearestNeabours[i * testData.size() +
        //             j]).label
        //             << "|"
        //             << ((Neabour)nearestNeabours[i * testData.size() +
        //             j]).index
        //             << "\t|";
        //     }
        //     std::cout << std::endl;
        // }
    }

  public:
    Knn(thrust::host_vector<LabelDataPoint<atributeCount>> &data) {
        thrust::device_vector<LabelDataPoint<atributeCount>> deviceData = data;
        calcNearestNeabours(deviceData, deviceData);
    }

    Knn(thrust::host_vector<LabelDataPoint<atributeCount>> &traingData,
        thrust::host_vector<LabelDataPoint<atributeCount>> &testData) {

        thrust::device_vector<LabelDataPoint<atributeCount>> deviceTraining =
            traingData;
        thrust::device_vector<LabelDataPoint<atributeCount>> deviceTest =
            testData;
        fillLabels(testData);
        calcNearestNeabours(deviceTraining, deviceTest);
    }

    void accuracy(Population<bool> &population,
                  thrust::device_vector<float> &accuracy) {
        thrust::fill(accuracy.begin(), accuracy.end(), 0.f);
        const dim3 threadsPerBlock = {32, 32, 1};

        if (labels.size() == 0) {
            const dim3 blocks = {(population.popSize() / threadsPerBlock.x) + 1,
                                 (population.popSize() / threadsPerBlock.y) + 1,
                                 1};

            sameAccuracyKernel<atributeCount, labelsCount, k>
                <<<blocks, threadsPerBlock>>>(
                    population.toDevicePopulation(),
                    thrust::raw_pointer_cast(&nearestNeabours[0]),
                    thrust::raw_pointer_cast(&accuracy[0]));

        } else {

            const dim3 blocks = {
                (population.popSize() / threadsPerBlock.x) + 1,
                ((unsigned int)labels.size() / threadsPerBlock.y) + 1, 1};
            accuracyKernel<atributeCount, labelsCount, k>
                <<<blocks, threadsPerBlock>>>(
                    population.toDevicePopulation(),
                    thrust::raw_pointer_cast(&nearestNeabours[0]),
                    thrust::raw_pointer_cast(&labels[0]),
                    thrust::raw_pointer_cast(&accuracy[0]), labels.size());
        }

        // thrust::transform(accuracy.begin(), accuracy.end(), accuracy.begin(),
        //                   [dataSize = dataSize] __device__(float acc) {
        //                       return acc / dataSize;
        //                   });
    }
};
