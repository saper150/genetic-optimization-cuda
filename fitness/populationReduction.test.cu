#include "../files/dataset.h"
#include "./KnnFitness.cuh"
#include <thrust/device_vector.h>
#include "./populationReduction.cuh"

#include "../catch.h"

TEST_CASE("sumPopulation should add all true gens", "[sumPopulation]") {
    {
        Population<bool> p(2, 5);
        thrust::device_vector<float> sums(p.popSize(), 0.f);
        sumPopulation<bool>(p, sums);
        REQUIRE(sums[0] == Approx(5.0f));
        REQUIRE(sums[1] == Approx(5.0f));
    }
    {
        Population<bool> p(2, 5);
        p.population[0] = false;
        p.population[1] = false;
        p.population[6] = false;
        thrust::device_vector<float> sums(p.popSize(), 0.f);
        sumPopulation<bool>(p, sums);

        REQUIRE(sums[0] == Approx(3.0f));
        REQUIRE(sums[1] == Approx(4.0f));
    }
}

TEST_CASE("sumPopulation should reset sums values", "[sumPopulation]"){
    Population<bool> p(2, 5);
    thrust::device_vector<float> sums(p.popSize(), 10.f);
    REQUIRE(sums[0] == Approx(10.f));
    sumPopulation<bool>(p, sums);
    REQUIRE(sums[0] == Approx(5.0f));
    REQUIRE(sums[1] == Approx(5.0f));
}
