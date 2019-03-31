#pragma once

#include <thrust/device_vector.h>
#include <sstream>
#include "../Knn/Knn.cuh"
#include "../lib/FloatArray.cuh"
#include "./fitnessTransform.cuh"
#include "./populationReduction.cuh"

template <int atributeCount, int labelsCount, int k>
struct KnnFitness {
 private:
  Knn<atributeCount, labelsCount, k> knn;
  thrust::device_vector<float> sums;
  int dataSize;
  void validateLabelsCount(
      const thrust::host_vector<LabelDataPoint<atributeCount>>& data) {
    const LabelDataPoint<atributeCount> max = *(thrust::max_element(
        data.begin(), data.end(),
        [](LabelDataPoint<atributeCount> a, LabelDataPoint<atributeCount> b) {
          return a.label < b.label;
        }));

    if ((max.label + 1) != labelsCount) {
      std::cout << (max.label + 1) << "|" << labelsCount << "|"
                << (int)((max.label + 1) != labelsCount) << "\n";
      std::ostringstream stringStream;
      stringStream << "labelsCount and actual labels does not match, expected "
                   << labelsCount << " got: " << (max.label + 1);
      throw stringStream.str().c_str();
    }
  }

 public:
  float alpha = 0.5f;
  float power = 3.f;
  KnnFitness(thrust::host_vector<LabelDataPoint<atributeCount>>& data)
      : knn(data), dataSize(data.size()) {
    validateLabelsCount(data);
  }

  void accuracy(Population<bool>& population,
                thrust::device_vector<float>& accuracy) {
    return knn.accuracy(population, accuracy);
  }
  void operator()(Population<bool>& population,
                  thrust::device_vector<float>& fitness) {
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
                        const float f = thrust::get<0>(t) / dataSize + 1.f -
                                        thrust::get<1>(t) / genSize;
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

template <int atributeCount, int labelsCount, int k>
class KnnFitnessNSGA {
  thrust::device_vector<float> accuracy;
  thrust::device_vector<float> sums;
  Knn<atributeCount, labelsCount, k> knn;
  int popSize;
 public:
  KnnFitnessNSGA(int popSize,
                 thrust::host_vector<LabelDataPoint<atributeCount>>& data)
      : accuracy(popSize), sums(popSize), knn(data), popSize(popSize) {}
  void operator()(Population<bool>& population,
                  thrust::device_vector<FloatArray<2>>& fitness) {
    knn.accuracy(population, accuracy);
    sumPopulation(population, sums);

    const auto begin = thrust::make_zip_iterator(
        thrust::make_tuple(accuracy.begin(), sums.begin()));

    const auto end = thrust::make_zip_iterator(
        thrust::make_tuple(accuracy.end(), sums.end()));
    thrust::transform(begin, end, fitness.begin(),
                      [] __device__(thrust::tuple<float, float> zip) {
                        FloatArray<2> f = {{zip.get<0>(), -zip.get<1>()}};
                        return f;
                      });
  }


  void exportResult(
      Population<bool>& population,
      thrust::host_vector<LabelDataPoint<atributeCount>>& trainData,
      thrust::host_vector<LabelDataPoint<atributeCount>>& testData,
      std::string file) {

    Knn<atributeCount, labelsCount, k> validateKnn(trainData, testData);
    thrust::device_vector<float> accuracy(popSize);
    validateKnn.accuracy(population, accuracy);
    sumPopulation(population, sums);
    thrust::device_vector<FloatArray<2>> fitness(popSize);

    const auto begin = thrust::make_zip_iterator(
        thrust::make_tuple(accuracy.begin(), sums.begin()));

    const auto end = thrust::make_zip_iterator(
        thrust::make_tuple(accuracy.end(), sums.end()));

    thrust::transform(begin, end, fitness.begin(),
                      [] __device__(thrust::tuple<float, float> zip) {
                        FloatArray<2> f = {{zip.get<0>(), zip.get<1>()}};
                        return f;
                      });

    std::ofstream myfile;
    myfile.open(file);
    myfile.clear();
    constexpr int cryteriaCount = 2;
    thrust::host_vector<FloatArray<cryteriaCount>> h = fitness;
    for (FloatArray<cryteriaCount> point : h) {
      for (int i = 0; i < cryteriaCount; i++) {
        myfile << point[i];
        if (i < cryteriaCount - 1) {
          myfile << ',';
        }
      }
      myfile << '\n';
    }
    myfile.close();
  }





};
