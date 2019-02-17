#pragma once
#include <curand.h>
#include <thrust/device_vector.h>
#include "../genetics/Population.cuh"

template <typename T>
struct Mutation {
  curandGenerator_t generator;
  float rate = 0;
  thrust::device_vector<float> rng;
  Mutation(int popSize, int genSize) : rng(popSize * genSize), rate(rate) {
    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
  }

  void mutate(Population<T>& p) {
    curandSetPseudoRandomGeneratorSeed(generator, time(0));
    curandGenerateUniform(generator, thrust::raw_pointer_cast(rng.data()),
                          rng.size());

    auto begin = thrust::make_zip_iterator(
        thrust::make_tuple(p.population.begin(), rng.begin()));
    auto end = thrust::make_zip_iterator(
        thrust::make_tuple(p.population.end(), rng.end()));

    thrust::transform(begin, end, p.population.begin(),
                      [rate = rate] __device__(thrust::tuple<bool, float> a) {
                        return a.get<1>() < rate ? !a.get<0>() : a.get<0>();
                      });
  }
};
