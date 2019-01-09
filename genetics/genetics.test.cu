#include "../catch.h"
#include "../fitness/BasicFitness.cuh"
#include "./GeneticAlgorithm.cuh"
#include <iostream>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>


TEST_CASE("GeneticAlgorithm", "genetic algorithm should be convergent") {
    BasicFitness<bool> basicFitness(100);

    GeneticAlgorithm<bool, BasicFitness<bool>> gen(200, basicFitness);
    thrust::device_vector<int> a(10);
    gen.maxFitness();
    gen.iterate();
    for (int i = 0; i < 1000; i++) {
        gen.iterate();
        if (gen.maxFitness() >= 100.f * 100.f * 100.f) {
            return;
        }
    }
    std::cout << gen.maxFitness() << std::endl;
    REQUIRE(true == false);
}
