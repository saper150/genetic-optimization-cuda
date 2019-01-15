#include <limits>
#include "../catch.h"
#include "../genetics/Population.cuh"
#include "./crossover.cuh"

TEST_CASE("crossover") {

  Population<bool> src1(2, 5);
  Population<bool> dest(2, 5);
  thrust::fill(dest.population.begin(), dest.population.end(), false);

  src1.population[0] = true;
  src1.population[1] = false;
  src1.population[2] = false;
  src1.population[3] = false;
  src1.population[4] = false;

  src1.population[5] = false;
  src1.population[6] = false;
  src1.population[7] = true;
  src1.population[8] = false;
  src1.population[9] = true;

  Population<bool> src2(2, 5);

  src2.population[0] = true;
  src2.population[1] = false;
  src2.population[2] = false;
  src2.population[3] = false;
  src2.population[4] = false;

  src2.population[5] = false;
  src2.population[6] = false;
  src2.population[7] = true;
  src2.population[8] = false;
  src2.population[9] = true;

  thrust::device_vector<thrust::tuple<int, int>> pairs(2);
  pairs[0] = thrust::make_tuple<int, int>(0, 3);
  pairs[1] = thrust::make_tuple<int, int>(1, 0);

  thrust::device_vector<int> rng(2);
  rng[0] = 0;
  rng[1] = 3;

  Crossover<bool> c;

  c.cross(src1, src2 , dest, pairs, rng);

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
