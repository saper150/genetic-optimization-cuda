#pragma once

#include "../Knn/Knn.cuh"
#include "./fitnessTransform.cuh"
#include "./populationReduction.cuh"
#include <sstream>
#include <thrust/device_vector.h>

template <int atributeCount, int labelsCount, int k> struct KnnFitness {
  private:
    Knn<atributeCount, labelsCount, k> knn;
    thrust::device_vector<float> sums;
    int dataSize;
    void validateLabelsCount(
        const thrust::host_vector<LabelDataPoint<atributeCount>> &data) {
        const LabelDataPoint<atributeCount> max = *(thrust::max_element(
            data.begin(), data.end(),
            [](LabelDataPoint<atributeCount> a,
               LabelDataPoint<atributeCount> b) { return a.label < b.label; }));

        if ((max.label + 1) != labelsCount) {
            std::cout << (max.label + 1) << "|" << labelsCount << "|"
                      << (int)((max.label + 1) != labelsCount) << "\n";
            std::ostringstream stringStream;
            stringStream
                << "labelsCount and actual labels does not match, expected "
                << labelsCount << " got: " << (max.label + 1);
            throw stringStream.str().c_str();
        }
    }

  public:
    float alpha = 0.5f;
    float power = 3.f;
    KnnFitness(thrust::host_vector<LabelDataPoint<atributeCount>> &data)
        : knn(data), dataSize(data.size()) {
        validateLabelsCount(data);
    }

    void accuracy(Population<bool> &population,
                  thrust::device_vector<float> &accuracy) {
        return knn.accuracy(population, accuracy);
    }
    void operator()(Population<bool> &population,
                    thrust::device_vector<float> &fitness) {
        sums.resize(population.popSize());
        sumPopulation(population, sums);
        knn.accuracy(population, fitness);
        const auto begin = thrust::make_zip_iterator(
            thrust::make_tuple(fitness.begin(), sums.begin()));
        const auto end = thrust::make_zip_iterator(
            thrust::make_tuple(fitness.end(), sums.end()));
        thrust::transform(begin, end, fitness.begin(),
                          [dataSize = dataSize, power = power,
                           genSize = population.genSize] __device__(auto t) {
                              if (thrust::get<1>(t) < k) {
                                  return 0.f;
                              }
                              const float f = thrust::get<0>(t) / dataSize +
                                              1.f - thrust::get<1>(t) / genSize;
                              return powf(f, power);
                          });
        //     [dataSize = dataSize] __device__(float f) { return f / dataSize;
        //     });        //     [dataSize = dataSize] __device__(float f) {
        //     return f / dataSize; });        //     [dataSize = dataSize]
        //     __device__(float f) { return f / dataSize; });        //// ////
        //     //     [dataSize = dataSize] __device__(float f) { return f /
        //     dataSize; });        //     [dataSize = dataSize]
        //     __device__(float f) { return f / dataSize; });
        //     //     [dataSize = dataSize] __device__(float f) { return f /
        //     dataSize; }); [dataSize = dataSize] __device__(float f) { return
        //     f / dataSize;
        //     }); [dataSize = dataSize] __device__(float f) { return f /
        //     dataSize;
        //     }); [dataSize = dataSize] __device__(float f) { return f /
        //     dataSize;
        //     }); [dataSize = dataSize] __device__(float f) { return f /
        //     dataSize;
        //     }); [dataSize = dataSize] __device__(float f) { return f /
        //     dataSize;
        //     });
        //     __device__(float f) { return f / dataSize; });
        //     __device__(float f) { return f / dataSize; });
        // return fitness;
    }
};
