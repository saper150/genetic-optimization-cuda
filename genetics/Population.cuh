#pragma once

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <iostream>

template <typename T>
struct DevicePopulation {
  T* population;
  const int popSize;
  const int genSize;
};

template <typename T>
struct Population {
  thrust::device_vector<T> population;
  int genSize;
  int popSize() const { return population.size() / genSize; }
  DevicePopulation<T> toDevicePopulation() {
    return {thrust::raw_pointer_cast(&population[0]), popSize(), genSize};
  }
  Population() {}
  Population(int popSize, int genSize)
      : genSize(genSize), population(popSize * genSize, true) {}
};

template <typename T>
T* getSpecimen(Population<T>& population, int index) {
  return thrust::raw_pointer_cast(population.population.data()) +
         (population.genSize * index);
}

template <typename T>
__device__ T* getSpecimen(const DevicePopulation<T>& population, int index) {
  return population.population + (index * population.genSize);
}

template <typename T>
void printPopulation(Population<T>& p, std::ostream& stream = std::cout) {
  thrust::host_vector<T> hostPopulation = p.population;

  for (int i = 0; i < p.popSize(); i++) {
    for (int j = 0; j < p.genSize; j++) {
      stream << hostPopulation[i * p.genSize + j] << '\t';
    }
    stream << '\n';
  }
  stream << '\n';
}
