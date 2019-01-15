#pragma once
#include "../genetics/Population.cuh"
#include "../genetics/Randomize.cuh"
#include "../sorting/sorting.cuh"
#include "./crossover.cuh"
#include "./selection.cuh"

#include <thrust/device_ptr.h>
#include "../crowdingDistance/crowdingDistance.cuh"

inline void sortLastGroup(thrust::host_vector<thrust::device_ptr<int>>& groups,
                          const thrust::device_vector<float>& distances) {
  thrust::sort(
      groups[groups.size() - 2], groups.back(),
      [distances = thrust::raw_pointer_cast(distances.data())] __device__(
          const int a, const int b) { return distances[a] > distances[b]; });
};

void printGroups(thrust::host_vector<thrust::device_ptr<int>> groups);


template <typename FitnessType, int cryteriaCount>
struct Genetics {
  Population<bool> p1;
  Population<bool> p2;
  Population<bool> p3;
  NonDominatedSorting<cryteriaCount> sorting;
  CrowdingDistance<cryteriaCount> crowdingDistance;
  Selection selection;
  IntRng selectionRng;
  IntRng crossoverRng;
  thrust::device_vector<FloatArray<cryteriaCount>> fitness;

  thrust::device_ptr<FloatArray<cryteriaCount>> p1Fitness;
  thrust::device_ptr<FloatArray<cryteriaCount>> p2Fitness;

  Genetics(int popSize, int genSize, FitnessType fitnesFunc)
      : p1(popSize, genSize),
        p2(popSize, genSize),
        p3(popSize, genSize),
        sorting(popSize * 2),
        crowdingDistance(popSize * 2),
        selection(popSize),
        selectionRng(popSize * 4, popSize),
        crossoverRng(popSize, genSize),
        fitness(popSize * 2) {
    p1Fitness = thrust::device_pointer_cast(fitness.data());
    p2Fitness =
        thrust::device_pointer_cast(fitness.data() + fitness.size() / 2);

    randomize(p1.population);
    randomize(p2.population);

    fitnesFunc(p1, p1Fitness);
    fitnesFunc(p2, p2Fitness);
    auto groups = sorting.sortHalf(fitness);

    const auto distances = crowdingDistance.calcDistances(groups, fitness);
    const auto generatedRng = selectionRng.generate();

    sortLastGroup(groups, distances);
    selection.select(groups, distances, generatedRng);

    Crossover<bool>().cross(p1, p2, selection.pairs, crossoverRng.generate());


  }

  void iterate() {}
};

