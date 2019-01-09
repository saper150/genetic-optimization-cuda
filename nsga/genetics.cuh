#pragma once
#include "../genetics/Population.cuh"
#include "../genetics/Randomize.cuh"
#include "../sorting/sorting.cuh"
#include "../sorting/sorting.cuh"

template <typename FitnessType, int cryteriaCount>
struct Genetics {
  Population<bool> p1;
  Population<bool> p2;
    NonDominatedSorting<cryteriaCount> sorting;
  Genetics(int popSize, int genSize, FitnessType fitnesFunc)
      : p1(popSize, genSize), p2(popSize, genSize), sorting(popSize) {
    randomize(p1.population);

    auto fitness = fitnesFunc(p1);
    const auto groups = sorting.sort(fitness);
    
    
  }

  void iterate() {}
};
