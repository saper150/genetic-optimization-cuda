#pragma once
#include "../genetics/Population.cuh"
#include "../genetics/Randomize.cuh"
#include "../sorting/sorting.cuh"
#include "./crossover.cuh"
#include "./mutation.cuh"
#include "./selection.cuh"

#include <thrust/device_ptr.h>
#include "../crowdingDistance/crowdingDistance.cuh"
#include "./PopFitness.cuh"

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
      // std::cout << (*groups[groups.size() - 2]).index << std::endl;
      std::cout << groups.size() << std::endl;
      std::cout << groups.back() - groups[groups.size() - 2] << std::endl;
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
  Population<bool> p3;
  NonDominatedSorting<cryteriaCount> sorting;
  CrowdingDistance<cryteriaCount> crowdingDistance;
  Selection<cryteriaCount> selection;
  IntRng selectionRng;
  IntRng crossoverRng;
  int popSize;
  int genSize;
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
        fitness(popSize * 2),
        popSize(popSize),
        genSize(genSize) {
    p1Fitness = thrust::device_pointer_cast(fitness.data());
    p2Fitness =
        thrust::device_pointer_cast(fitness.data() + fitness.size() / 2);

    Mutation<bool> mutation(popSize, genSize);
    mutation.rate = 0.02f;
    randomize(p1.population);
    randomize(p2.population);

    fitnesFunc(p1, p1Fitness);
    fitnesFunc(p2, p2Fitness);

    thrust::device_vector<PopFitness<cryteriaCount>> fitnessPop(fitness.size());
    copyPopFitness(thrust::raw_pointer_cast(fitnessPop.data()), p1Fitness.get(),
                   p1);
    copyPopFitness(thrust::raw_pointer_cast(fitnessPop.data() + popSize),
                   p2Fitness.get(), p2);

    thrust::host_vector<thrust::device_ptr<PopFitness<cryteriaCount>>> groups =
        sorting.sortHalfPop(fitnessPop);
    cudaDeviceSynchronize();
    auto distances = crowdingDistance.calcDistancesPop(groups);
    cudaDeviceSynchronize();
    sortLastGroupPop<cryteriaCount>(groups, distances);

    Population<bool> parentPop(popSize, genSize);

    // const dim3 threadsPerBlock = {32, 32, 1};
    // const dim3 blocks = {(parentPop.popSize() / threadsPerBlock.x) + 1,
    //                      (parentPop.popSize() / threadsPerBlock.y) + 1, 1};
    // copyPopulation<cryteriaCount><<<blocks, threadsPerBlock>>>(
    //     thrust::raw_pointer_cast(fitnessPop.data()),
    //     parentPop.toDevicePopulation());

    copySpeciments(fitnessPop, parentPop);

    auto generatedRng = selectionRng.generate();
    cudaDeviceSynchronize();
    selection.selectPop(groups, distances, generatedRng);
    cudaDeviceSynchronize();
    Crossover<bool>().crossPop(p3, selection.pairsPop, crossoverRng.generate());
    cudaDeviceSynchronize();
    // printPopulation(p3, std::cout);
    std::cout << std::endl;

    // for (int i = 0; i < 0; i++) {
    // //   fitnesFunc(p3, p2Fitness);

    //   copyPopFitness(thrust::raw_pointer_cast(fitnessPop.data() + popSize),
    //                  p2Fitness.get(), p3);

    //   groups = sorting.sortHalfPop(fitnessPop);
    //   distances = crowdingDistance.calcDistancesPop(groups);
    //   sortLastGroupPop<cryteriaCount>(groups, distances);

    //   {
    //     const dim3 threadsPerBlock = {32, 32, 1};
    //     const dim3 blocks = {(parentPop.popSize() / threadsPerBlock.x) + 1,
    //                          (parentPop.popSize() / threadsPerBlock.y) + 1, 1};
    //     copyPopulation<cryteriaCount><<<blocks, threadsPerBlock>>>(
    //         thrust::raw_pointer_cast(fitnessPop.data()),
    //         parentPop.toDevicePopulation());

    //     copySpeciments(fitnessPop, parentPop);
    //   }

    //   generatedRng = selectionRng.generate();

    //   selection.selectPop(groups, distances, generatedRng);
    //   Crossover<bool>().crossPop(p1, selection.pairsPop,
    //                              crossoverRng.generate());
    //   mutation.mutate(p1);
    //   std::swap(p1, p3);
    // }

    // printPopulation(p1, std::cout);

    // fitnesFunc(p3, p2Fitness);
    // copyPopFitness(thrust::raw_pointer_cast(fitnessPop.data() + popSize),
    //                p2Fitness.get(), p3);
    // groups = sorting.sortHalfPop(fitnessPop);
    // distances = crowdingDistance.calcDistancesPop(groups);
    // sortLastGroupPop<cryteriaCount>(groups, distances);
    // generatedRng = selectionRng.generate();

    // thrust::copy_n(groups.front(), popSize, hostFitness.begin());

    // for (int i = 0; i < popSize; i++) {
    //   thrust::copy_n(hostFitness[i].specimen, genSize,
    //                  p2.population.begin() + genSize * i);
    //   hostFitness[i].specimen = getSpecimen(p3, i);
    //   groups.front()[i] = hostFitness[i];
    // }

    // selection.selectPop(groups, distances, generatedRng);
    // Crossover<bool>().crossPop(p3, selection.pairsPop,
    // crossoverRng.generate());

    // Crossover<bool>().cross(p1, p2, p3, selection.pairs,
    //                         crossoverRng.generate());

    // fitnesFunc(p3, p1Fitness);

    // fitnesFunc(p3, p1Fitness)
  }

  // void copy(thrust::host_vector<thrust::device_ptr<int>> groups) {
  //   thrust::for_each_n(thrust::counting_iterator<int>(0), popSize,
  //                      [g = groups[0].get(), p1 = p1.toDevicePopulation(),
  //                       p2 = p2.toDevicePopulation(),
  //                       o = p3.toDevicePopulation(),
  //                       popSize = popSize
  //                       ] __device__(int i) {

  //                       DevicePopulation<bool> parent = g[i] >= popSize ? p2
  //                       : p1;

  //                        thrust::copy(getSpecimen(parent, g[i]),
  //                                     getSpecimen(parent, g[i]) + p1.genSize,
  //                                     getSpecimen(o, i));
  //                        if (i == 0) {
  //                          cudaDeviceSynchronize();
  //                        }
  //                      });
  // }

  void copySpeciments(
      thrust::device_vector<PopFitness<cryteriaCount>>& fitnessPop,
      Population<bool>& p) {
    thrust::for_each_n(thrust::make_counting_iterator<int>(0), popSize,
                       [g = thrust::raw_pointer_cast(fitnessPop.data()),
                        pop = p.toDevicePopulation()] __device__(int i) {
                         g[i].specimen = getSpecimen(pop, i);
                       });
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

  void iterate() {}
};
