
#include "../DeviceArray.h"
#include "./genetics/Population.cuh"
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
                      DeviceArray<LabelDataPoint<atributeCount>> *teaching,
                      DeviceArray<LabelDataPoint<atributeCount>> *test) {
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    const int idy = threadIdx.y + blockDim.y * blockIdx.y;
    if (idx >= teaching.size || idy >= test.size) {
        return;
    }
    distances[idy * size + idx] = distanceSquared(teaching[idy], test[idx]);
    neabours[idy * size + idx] = {teaching[idx].label, idx};
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
                               float *accuracy) {
    const int populationId = threadIdx.x + blockDim.x * blockIdx.x;
    const int rowId = threadIdx.y + blockDim.y * blockIdx.y;

    if (populationId >= population.popSize || rowId >= population.genSize) {
        return;
    }

    const auto specimen = getSpecimen(population, populationId);
    const auto neabours = nearestNeabours + rowId * population.genSize;

    int foundCount = 0;
    int labelsCounts[labelsCount] = {0};

    for (size_t i = 1; i < population.genSize; i++) {
        if (specimen[neabours[i].index]) {
            labelsCounts[neabours[i].label]++;
            foundCount++;
            if (foundCount >= k) {
                break;
            }
        }
    }

    const auto voutedLabel = thrust::max_element(thrust::seq, labelsCounts,
                                                 labelsCounts + labelsCount) -
                             labelsCount;
    if (voutedLabel == labels[rowId]) {
        atomicAdd(&accuracy[populationId], 1.f);
    }
};

template <typename T> struct GetLabel : public thrust::unary_function<T, int> {
    __host__ __device__ float operator()(T a) const { return a.label; }
};

template <int atributeCount, int labelsCount, int K> class Knn {
  private:
    thrust::device_vector<Neabour> nearestNeabours;
    thrust::device_vector<int> labels(0);

    fillLabels(const thrust::host_vector<LabelDataPoint<atributeCount>> &data) {
        labels.resize(data.size());
        const auto beginIter =
            thrust::make_transform_iterator(testData.begin(), GetLabel());
        const auto endIter =
            thrust::make_transform_iterator(testData.end(), GetLabel());

        thrust::copy(beginIter, endIter, labels.begin());
    }

    calcNearestNeabours(
        const thrust::host_vector<LabelDataPoint<atributeCount>> &traingData,
        const thrust::host_vector<LabelDataPoint<atributeCount>> &testData) {
        const dim3 perBlock = {32, 32, 1};
        const dim3 blocks = {(unsigned int)traingData.size() / perBlock.x + 1,
                             (unsigned int)testData.size() / perBlock.y + 1, 1};

        thrust::device_vector<LabelDataPoint<atributeCount>> deviceData = data;

        thrust::device_vector<float> distances(traingData.size() *
                                               testData.size());

        nearestNeaboursKernel<<<blocks, perBlock>>>(
            thrust::raw_pointer_cast(&distances[0]),
            thrust::raw_pointer_cast(&nearestNeabours[0]),
            toDeviceArray(trainData), toDeviceArray(testData));

        thrust::host_vector<cudaStream_t> streams(data.size());
        for (size_t i = 0; i < data.size(); i++) {
            cudaStreamCreate(&streams[i]);
            thrust::sort_by_key(thrust::cuda::par.on(streams[i]),
                                distances.begin() + i * data.size(),
                                distances.begin() + (i + 1) * data.size(),
                                nearestNeabours.begin() + i * data.size());
            cudaStreamDestroy(streams[i]);
        }
    }

  public:
    Knn(const thrust::host_vector<LabelDataPoint<atributeCount>> &data) {
        calculateNearestNeabours(data, data);
    }

    Knn(const thrust::host_vector<LabelDataPoint<atributeCount>> &traingData,
        const thrust::host_vector<LabelDataPoint<atributeCount>> &testData) {

        fillLabels(testData);
        calculateNearestNeabours(trainData, testData);
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
            const dim3 blocks = {(population.popSize() / threadsPerBlock.x) + 1,
                                 (labels.size() / threadsPerBlock.y) + 1, 1};
            accuracyKernel<atributeCount, labelsCount, k>
                <<<blocks, threadsPerBlock>>>(
                    population.toDevicePopulation(),
                    thrust::raw_pointer_cast(&nearestNeabours[0]),
                    thrust::raw_pointer_cast(&labels[0]),
                    thrust::raw_pointer_cast(&accuracy[0]));
        }

        // thrust::transform(accuracy.begin(), accuracy.end(), accuracy.begin(),
        //                   [dataSize = dataSize] __device__(float acc) {
        //                       return acc / dataSize;
        //                   });
    }
};
