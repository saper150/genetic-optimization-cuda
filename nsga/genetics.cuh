#pragma once
#include "../genetics/Population.cuh"
#include "../genetics/Randomize.cuh"
#include "../sorting/sorting.cuh"
#include "./crossover.cuh"
#include "./mutation.cuh"
#include "./selection.cuh"

#include <thrust/device_ptr.h>
#include <fstream>
#include "../crowdingDistance/crowdingDistance.cuh"
#include "./PopFitness.cuh"

template <int c>
void printGroupss(
    thrust::host_vector<thrust::device_ptr<PopFitness<c>>> groups) {
  std::cout << "size" << groups.size() << " " << groups[0] - groups[1] << '\n';

  for (size_t i = 0; i < groups.size() - 1; i++) {
    thrust::device_ptr<PopFitness<c>>& begin = groups[i];
    thrust::device_ptr<PopFitness<c>>& end = groups[i + 1];
    const int groupSize = end - begin;
    std::cout << "group " << i << ", group size:" << groupSize << '\n';
    for (int j = 0; j < groupSize; j++) {
      PopFitness<c> cc = *(begin + j);
      std::cout << "(" << cc.fitness.data[0] << ", " << cc.fitness.data[1]
                << ", " << cc.fitness.data[1] + cc.fitness.data[0] << ")"
                << '\n';
    }
  }
}

inline void sortLastGroup(thrust::host_vector<thrust::device_ptr<int>>& groups,
                          const thrust::device_vector<float>& distances) {
  thrust::sort(
      groups[groups.size() - 2], groups.back(),
      [distances = thrust::raw_pointer_cast(distances.data())] __device__(
          const int a, const int b) { return distances[a] > distances[b]; });
};

template <int cryteriaCount>
void sortLastGroupPop(
    thrust::host_vector<thrust::device_ptr<PopFitness<cryteriaCount>>>& groups,
    const thrust::device_vector<float>& distances) {
  thrust::sort(
      groups[groups.size() - 2], groups.back(),
      [distances = thrust::raw_pointer_cast(distances.data())] __device__(
          const PopFitness<cryteriaCount> a,
          const PopFitness<cryteriaCount> b) {
        return distances[a.index] > distances[b.index];
      });
};

template <int cryteriaCount>
__global__ void copyPopulation(PopFitness<cryteriaCount>* g,
                               DevicePopulation<bool> dest) {
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;
  if (idx >= dest.popSize || idy >= dest.genSize) {
    return;
  }
  getSpecimen(dest, idx)[idy] = g[idx].specimen[idy];
};

void printGroups(thrust::host_vector<thrust::device_ptr<int>> groups);

template <typename FitnessType, int cryteriaCount>
struct Genetics {
  Population<bool> p1;
  Population<bool> p2;
  NonDominatedSorting<cryteriaCount> sorting;
  CrowdingDistance<cryteriaCount> crowdingDistance;
  Selection<cryteriaCount> selection;
  IntRng selectionRng;
  IntRng crossoverRng;
  int popSize;
  int genSize;
  Mutation<bool> mutation;
  thrust::device_vector<FloatArray<cryteriaCount>> fitness;
  thrust::host_vector<thrust::device_ptr<PopFitness<cryteriaCount>>> groups;
  thrust::device_vector<PopFitness<cryteriaCount>> fitnessPop;
  Genetics(int popSize, int genSize, FitnessType* fitnesFunc)
      : p1(popSize, genSize),
        p2(popSize, genSize),
        sorting(popSize),
        crowdingDistance(popSize),
        selection(popSize),
        selectionRng(popSize * 4, popSize),
        crossoverRng(popSize, genSize),
        fitness(popSize),
        popSize(popSize),
        genSize(genSize),
        mutation(popSize, genSize),
        fitnessPop(popSize) {
    mutation.rate = 0.01f;
    randomize(p1.population);

    for (int i = 0; i < 150; i++) {
      std::cout << i << std::endl;
      (*fitnesFunc)(p1, fitness);
      copyPopFitness(thrust::raw_pointer_cast(fitnessPop.data()),
                     fitness.data().get(), p1);

      groups = sorting.sortHalfPop(fitnessPop);
      auto& distances = crowdingDistance.calcDistancesPop(groups);

      auto& generatedRng = selectionRng.generate();
      selection.selectPop(groups, distances, generatedRng);

      Crossover<bool>().crossPop(p2, selection.pairsPop,
                                 crossoverRng.generate());
      mutation.mutate(p2);
      std::swap(p1, p2);
    }
    printGroupss(groups);
    // printPopulation(p1);
  }

  void copyPopFitness(PopFitness<cryteriaCount>* dest,
                      FloatArray<cryteriaCount>* fromF,
                      Population<bool>& from) {
    thrust::for_each_n(thrust::counting_iterator<int>(0), popSize,
                       [dest = dest, fromF = fromF,
                        from = from.toDevicePopulation()] __device__(int i) {
                         dest[i] = {i, getSpecimen(from, i), fromF[i]};
                       });
  }

  void exportResult(std::string file) {
    std::ofstream myfile;
    myfile.open(file);
    myfile.clear();
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
