#include <limits>
#include "../catch.h"
#include "../genetics/Population.cuh"
#include "./crossover.cuh"

TEST_CASE("crossover") {
  // REQUIRE(0 == 1);
  Population<bool> src(2, 5);
  Population<bool> dest(2, 5);
  thrust::fill(dest.population.begin(), dest.population.end(), false);

  src.population[0] = true;
  src.population[1] = false;
  src.population[2] = false;
  src.population[3] = false;
  src.population[4] = false;

  src.population[5] = false;
  src.population[6] = false;
  src.population[7] = true;
  src.population[8] = false;
  src.population[9] = true;

  thrust::device_vector<thrust::tuple<int, int>> pairs(2);
  pairs[0] = thrust::make_tuple<int, int>(0, 1);
  pairs[1] = thrust::make_tuple<int, int>(1, 0);

  thrust::device_vector<int> rng(2);
  rng[0] = 0;
  rng[1] = 3;

  Crossover<bool> c;

  c.cross(src, dest, pairs, rng);

  REQUIRE(dest.population[0] == false);
  REQUIRE(dest.population[1] == false);
  REQUIRE(dest.population[2] == true);
  REQUIRE(dest.population[3] == false);
  REQUIRE(dest.population[4] == true);

  REQUIRE(dest.population[5] == false);
  REQUIRE(dest.population[6] == false);
  REQUIRE(dest.population[7] == true);
  REQUIRE(dest.population[8] == false);
  REQUIRE(dest.population[9] == false);

}
