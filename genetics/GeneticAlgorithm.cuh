#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "./Population.cuh"
#include "./Selection.h"

#include "./CrossOver.cuh"
#include "./Mutation.cuh"

#include "../Performance/Performance.h"
#include "./Randomize.cuh"


template <typename T, typename Selection, typename CrossOver, typename Mutation>
__global__ void geneticKernel(DevicePopulation<T> populationA,
                              DevicePopulation<T> populationB,
                              const float *fitness, Selection selection,
                              CrossOver crossover, Mutation mutation) {
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;

    const auto individualA = selection(populationA, fitness);
    const auto individualB = selection(populationA, fitness);
    crossover(populationA.genSize, getSpecimen(populationB, idx), individualA,
              individualB);
    mutation(getSpecimen(populationB, idx), populationB.genSize);
}

template <typename T, typename Fitness> class GeneticAlgorithm {
  private:
    Population<T> populationA;
    Population<T> populationB;
    Fitness fitness;
    // SinglePointCrossover<T> crossover;
    MultiPointCrossover<T, 4> crossover;
    RouletteSelection<T> selection;
    Mutation<T> mutation = {0.002f};
    int popSize;
    thrust::device_vector<float> fitnessValues;

  public:
    GeneticAlgorithm(const int popSize, Fitness fitness)
        : popSize(popSize), fitness(fitness), fitnessValues(popSize) {

        Performance::mesure("alocate population", [&]() {
            populationA = Population<T>(popSize, fitness.genSize());
            populationB = Population<T>(popSize, fitness.genSize());
        });

        Performance::mesure("randomize population",
                            [&]() { randomize(populationA.population); });
    };

    void iterate() {
        fitness(populationA, fitnessValues);
        selection.iteration(populationA, fitnessValues);

        geneticKernel<T><<<1, popSize>>>(
            populationA.toDevicePopulation(), populationB.toDevicePopulation(),
            thrust::raw_pointer_cast(&fitnessValues[0]), selection, crossover,
            mutation);
        std::swap(populationA, populationB);
    }

    float maxFitness() {
        return *(thrust::max_element(fitnessValues.begin(), fitnessValues.end()));
    }
};
