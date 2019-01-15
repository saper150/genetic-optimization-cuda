#include <limits>
#include "../catch.h"
#include "../sorting/sorting.cuh"
#include "./selection.cuh"
#include "../crowdingDistance/crowdingDistance.cuh"

TEST_CASE("fillFronts should fill fronts") {
  constexpr int cryteriaCount = 2;
  thrust::device_vector<FloatArray<cryteriaCount>> fitnesses(5);
  fitnesses[0] = {0, 0};
  fitnesses[1] = {100, 100};
  fitnesses[2] = {8, 11};
  fitnesses[3] = {7, 12};
  fitnesses[4] = {1, 1};

  NonDominatedSorting<cryteriaCount> s(5);
  thrust::host_vector<thrust::device_ptr<int>> groups = s.sort(fitnesses);
  thrust::device_vector<int> fronts(5);

  fillFronts(groups, fronts);

  REQUIRE(fronts[0] == 3);
  REQUIRE(fronts[1] == 0);
  REQUIRE(fronts[2] == 1);
  REQUIRE(fronts[3] == 1);
  REQUIRE(fronts[4] == 2);

}


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

TEST_CASE("selection") {
  constexpr int cryteriaCount = 2;
  thrust::device_vector<FloatArray<cryteriaCount>> fitnesses(6);
  fitnesses[0] = {15, 7};
  fitnesses[1] = {8, 15};
  fitnesses[2] = {11, 9};
  fitnesses[3] = {9, 13};
  fitnesses[4] = {2, 2};
  fitnesses[5] = {3, 1};


  NonDominatedSorting<cryteriaCount> s(6);
  thrust::host_vector<thrust::device_ptr<int>> groups = s.sort(fitnesses);

  CrowdingDistance<cryteriaCount> crowding(6);

  crowding.calcDistances(groups, fitnesses);

  Selection ss(6);

  thrust::device_vector<int> fakeRng(5* 4);
  fakeRng[0] = 0;
  fakeRng[1] = 4;
  fakeRng[2] = 1;
  fakeRng[3] = 3;

  fakeRng[4] = 5;
  fakeRng[5] = 3;
  fakeRng[6] = 4;
  fakeRng[7] = 3;

  groups[0][0] = 0;
  groups[0][1] = 1;
  groups[0][2] = 2;
  groups[0][3] = 3;
  groups[0][4] = 4;
  groups[0][5] = 4;

  ss.select(groups, crowding.crowdDistances, fakeRng);
  thrust::tuple<int, int> f1 = ss.pairs[0];
  REQUIRE(f1 == thrust::make_tuple<int, int>(0, 1));
  thrust::tuple<int, int> f2 = ss.pairs[1];
  REQUIRE(f2 == thrust::make_tuple<int, int>(3, 3));

}
