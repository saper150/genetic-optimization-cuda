#pragma once
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

template <int cryteriaCount>
struct CrowdingDistance {
  thrust::device_vector<float> crowdDistances;
  FloatArray<cryteriaCount>* fitnessesPointer;
  CrowdingDistance(int popSize) : crowdDistances(popSize) {}

  thrust::device_vector<float>& calcDistances(
      const thrust::host_vector<thrust::device_ptr<int>>& groups,
      thrust::device_vector<FloatArray<cryteriaCount>>& fitnesses) {
    thrust::fill(crowdDistances.begin(), crowdDistances.end(), 0.f);
    fitnessesPointer = thrust::raw_pointer_cast(fitnesses.data());
    for (int i = 0; i < groups.size() - 1; i++) {
      auto begin = groups[i];
      auto end = groups[i + 1];
      for (int j = 0; j < cryteriaCount; j++) {
        sortByCryterium(begin, end, j);
      }
    }
    return crowdDistances;
  }

    thrust::device_vector<float>& calcDistancesPop(
      const thrust::host_vector<thrust::device_ptr<PopFitness<cryteriaCount>>>& groups) {
    thrust::fill(crowdDistances.begin(), crowdDistances.end(), 0.f);
    for (int i = 0; i < groups.size() - 1; i++) {
      auto begin = groups[i];
      auto end = groups[i + 1];
      for (int j = 0; j < cryteriaCount; j++) {
        sortByCryteriumPop(begin, end, j);
      }
    }
    return crowdDistances;
  }

  void sortByCryteriumPop(thrust::device_ptr<PopFitness<cryteriaCount>> begin,
                       thrust::device_ptr<PopFitness<cryteriaCount>> end,
                       const int cryterium) {
    thrust::sort(begin, end,
                 [cryterium = cryterium] __device__(
                     PopFitness<cryteriaCount> a, PopFitness<cryteriaCount> b) {
                   return a.fitness[cryterium] < b.fitness[cryterium];
                 });

    PopFitness<cryteriaCount> min = *begin;
    // cudaMemcpy(&min, begin.get(), sizeof(min), cudaMemcpyDeviceToHost);

    PopFitness<cryteriaCount> max = *(begin + (end - begin - 1));
    // cudaMemcpy(&max, begin.get() + (end - begin - 1), sizeof(max),
    //            cudaMemcpyDeviceToHost);

    // cudaMemcpy(thrust::raw_pointer_cast(crowdDistances[min]), , sizeof(min),
    // cudaMemcpyDeviceToHost);

    crowdDistances[min.index] = std::numeric_limits<float>::infinity();
    crowdDistances[max.index] = std::numeric_limits<float>::infinity();

    const int size = end - begin - 2;
    if (size <= 0) {
      return;
    }

    const auto foreachFunc =
        [sortedGroup = begin.get() + 1,
         cryterium = cryterium, size = size,
         distances = thrust::raw_pointer_cast(
             crowdDistances.data())] __device__(const int idx) {
          const float toAdd =
              (sortedGroup[idx + 1].fitness[cryterium] -
               sortedGroup[idx - 1].fitness[cryterium]) /
              (sortedGroup[size].fitness[cryterium] -
               sortedGroup[-1].fitness[cryterium]);
          atomicAdd(distances + sortedGroup[idx].index, toAdd);
        };

    thrust::for_each_n(thrust::make_counting_iterator<int>(0), size,
                       foreachFunc);
  }


  void sortByCryterium(thrust::device_ptr<int> begin,
                       thrust::device_ptr<int> end,
                       const int cryterium) {
    thrust::sort(begin, end,
                 [fitness = fitnessesPointer, cryterium = cryterium] __device__(
                     int a, int b) {
                   return fitness[a][cryterium] < fitness[b][cryterium];
                 });

    int min = *begin;
    // cudaMemcpy(&min, begin.get(), sizeof(min), cudaMemcpyDeviceToHost);

    int max = *(begin + (end - begin - 1));
    // cudaMemcpy(&max, begin.get() + (end - begin - 1), sizeof(max),
    //            cudaMemcpyDeviceToHost);

    // cudaMemcpy(thrust::raw_pointer_cast(crowdDistances[min]), , sizeof(min),
    // cudaMemcpyDeviceToHost);

    crowdDistances[min] = std::numeric_limits<float>::infinity();
    crowdDistances[max] = std::numeric_limits<float>::infinity();

    const int size = end - begin - 2;
    if (size <= 0) {
      return;
    }

    const auto foreachFunc =
        [fitnessesPointer = fitnessesPointer, sortedGroup = begin.get() + 1,
         cryterium = cryterium, size = size,
         distances = thrust::raw_pointer_cast(
             crowdDistances.data())] __device__(const int idx) {
          const float toAdd =
              (fitnessesPointer[sortedGroup[idx + 1]][cryterium] -
               fitnessesPointer[sortedGroup[idx - 1]][cryterium]) /
              (fitnessesPointer[sortedGroup[size]][cryterium] -
               fitnessesPointer[sortedGroup[-1]][cryterium]);
          atomicAdd(distances + sortedGroup[idx], toAdd);
        };

    thrust::for_each_n(thrust::make_counting_iterator<int>(0), size,
                       foreachFunc);
  }
};
