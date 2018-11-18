#include "../catch.h"
#include "./sorting.cuh"
#include <thrust/device_vector.h>

TEST_CASE("should sort sort fitness") {
    constexpr int cryteriaCount = 2;
    thrust::device_vector<FloatArray<cryteriaCount>> fitnesses(4);
    fitnesses[0] = {0, 0};
    fitnesses[1] = {100, 100};
    fitnesses[2] = {150, 20};
    fitnesses[3] = {160, 10};

    NonDominatedSorting<cryteriaCount> s(4);
    thrust::device_vector<int> sorted(4);
    s.sort(fitnesses, sorted);

    REQUIRE(sorted[0] == 1);
    REQUIRE(sorted[3] == 0);
}

TEST_CASE("isDominated") {
    {
        FloatArray<2> f = {{10, 10}};
        FloatArray<2> f2 = {{0, 0}};
        REQUIRE(isDominating(f, f2) == true);
    }

    {
        FloatArray<2> f = {{0, 0}};
        FloatArray<2> f2 = {{10, 10}};
        REQUIRE(isDominating(f, f2) == false);
    }

    {
        FloatArray<2> f = {{9, 10}};
        FloatArray<2> f2 = {{10, 9}};
        REQUIRE(isDominating(f, f2) == false);
    }

    {
        FloatArray<2> f = {{5, 20}};
        FloatArray<2> f2 = {{11, 9}};
        REQUIRE(isDominating(f, f2) == false);
    }
}