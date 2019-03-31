#pragma once

#include <thrust/functional.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/transform.h>
#include "../genetics/Population.cuh"

struct ToFloat : public thrust::unary_function<bool, float> {
  __host__ __device__ float operator()(bool a) const {
    return static_cast<float>(a);
  }
};

using namespace thrust::placeholders;
// template <typename T>
// void getReductionLevels(const Population<T> &population,
//                         thrust::device_vector<float> &reductionsLevels) {

//     const auto keysBegin = thrust::make_transform_iterator(
//         thrust::counting_iterator<int>(0), _1 / population.genSize);

//     const auto keysEnd = thrust::make_transform_iterator(
//         thrust::counting_iterator<int>(100), _1 / population.genSize);

//     const auto dataIterator = thrust::make_transform_iterator(
//         population.population.begin(), ToFloat());

//     thrust::reduce_by_key(thrust::device, keysBegin, keysEnd, dataIterator,
//                           thrust::discard_iterator<int>(),
//                           reductionsLevels.begin());

//     thrust::transform(reductionsLevels.begin(), reductionsLevels.end(),
//                       reductionsLevels.begin(),
//                       [genSize = population.genSize] __device__(float val) {
//                           return 1.f - (val / genSize);
//                       });

//     // thrust::reduce_by_key(
//     //     thrust::device,
//     //     thrust::make_transform_iterator(thrust::counting_iterator<int>(0),
//     //                                     _1 / K),
//     //     thrust::make_transform_iterator(thrust::counting_iterator<int>(N *
//     //     K),
//     //                                     _1 / K),
//     //     data.begin(), thrust::discard_iterator<int>(), sums.begin());
//     // thrust::reduce_by_key(
//     //     thrust::device,
//     //     thrust::make_transform_iterator(thrust::counting_iterator<int>(0),
//     //                                     _1 / population.genSize),
//     //     thrust::make_transform_iterator(
//     //         thrust::counting_iterator<int>(population.population.size()),
//     //         _1 / population.genSize),
//     //     population.population.begin(), thrust::discard_iterator<int>(),
//     //     reductionsLevels.begin());
// }

template <typename T>
void sumPopulation(const Population<T>& population,
                   thrust::device_vector<float>& sums) {
  const auto keysBegin = thrust::make_transform_iterator(
      thrust::counting_iterator<int>(0), _1 / population.genSize);

  const auto keysEnd = thrust::make_transform_iterator(
      thrust::counting_iterator<int>(population.population.size()),
      _1 / population.genSize);

  const auto dataIterator =
      thrust::make_transform_iterator(population.population.begin(), ToFloat());

  thrust::reduce_by_key(thrust::device, keysBegin, keysEnd, dataIterator,
                        thrust::discard_iterator<int>(), sums.begin());

  // thrust::transform(reductionsLevels.begin(), reductionsLevels.end(),
  //                   reductionsLevels.begin(),
  //                   [genSize = population.genSize] __device__(float val) {
  //                       return 1.f - (val / genSize);
  //                   });
}