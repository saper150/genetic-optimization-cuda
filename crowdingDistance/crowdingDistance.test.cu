#include <limits>
#include "../catch.h"
#include "../sorting/sorting.cuh"
#include "./crowdingDistance.cuh"

void printGroups(thrust::host_vector<thrust::device_ptr<int>> groups) {
  for (size_t i = 0; i < groups.size() - 1; i++) {
    std::cout << "group " << i << '\n';
    auto begin = *groups[i];
    auto end = *groups[i + 1];
    for (int j = 0; j < &end - &begin; j++) {
      std::cout << (&begin)[j] << '\n';
    }
  }
}

TEST_CASE("Crowding Distances calculations") {
  constexpr int cryteriaCount = 2;
  thrust::device_vector<FloatArray<cryteriaCount>> fitnesses(6);
  fitnesses[0] = {15, 7};
  fitnesses[1] = {8, 15};
  fitnesses[2] = {11, 9};
  fitnesses[3] = {9, 13};
  fitnesses[4] = {2, 2};
  fitnesses[5] = {3, 1};

  NonDominatedSorting<cryteriaCount> s(6);
  auto groups = s.sort(fitnesses);

  CrowdingDistance<cryteriaCount> crowding(6);
  crowding.calcDistances(groups, fitnesses);

  REQUIRE(crowding.crowdDistances[0] == std::numeric_limits<float>::infinity());
  REQUIRE(crowding.crowdDistances[1] == std::numeric_limits<float>::infinity());
  REQUIRE(crowding.crowdDistances[3] == .3f / .7f + .6f / .8f);

  REQUIRE(crowding.crowdDistances[4] == std::numeric_limits<float>::infinity());
  REQUIRE(crowding.crowdDistances[5] == std::numeric_limits<float>::infinity());
}
