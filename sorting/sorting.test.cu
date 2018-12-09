#include "../catch.h"
#include "./sorting.cuh"
#include <sstream>
#include <thrust/device_vector.h>

class GroupContains : public Catch::MatcherBase<int> {
  DominanceGroups groups;
  int group;

public:
  GroupContains(DominanceGroups groups, int group)
      : groups(groups), group(group) {}

  // Performs the test for this matcher
  virtual bool match(int const &nidle) const override {
    auto begin = groups[group];
    auto end = groups[group + 1];
    for (int i = 0; i < end - begin; i++) {
      if (begin[i] == nidle) {
        return true;
      }
    }
    return false;
  }

  // Produces a string describing what this matcher does. It should
  // include any provided data (the begin/ end in this case) and
  // be written as if it were stating a fact (in the output it will be
  // preceded by the value under test).
  virtual std::string describe() const {
    std::ostringstream ss;
    ss << "is in group number " << group;
    return ss.str();
  }
};

// The builder function
// inline GroupContains IsBetween(int begin, int end) { return IntRange(begin,
// end); }

TEST_CASE("should sort sort fitness") {
  constexpr int cryteriaCount = 2;
  thrust::device_vector<FloatArray<cryteriaCount>> fitnesses(5);
  fitnesses[0] = {0, 0};
  fitnesses[1] = {100, 100};
  fitnesses[2] = {8, 11};
  fitnesses[3] = {7, 12};
  fitnesses[4] = {1, 1};

  NonDominatedSorting<cryteriaCount> s(5);
  thrust::device_vector<int> sorted(5);
  auto groups = s.sort(fitnesses, sorted);

  REQUIRE(groups.size() == 5);

  REQUIRE(groups[1] - groups[0] == 1); // size of first group
  REQUIRE(groups[2] - groups[1] == 2); // size of second group
  REQUIRE(groups[3] - groups[2] == 1); // size of third group
  REQUIRE(groups[4] - groups[3] == 1); // size of forth group

  CHECK_THAT(1, GroupContains(groups, 0));

  CHECK_THAT(3, GroupContains(groups, 1));
  CHECK_THAT(2, GroupContains(groups, 1));

  CHECK_THAT(4, GroupContains(groups, 2));

  CHECK_THAT(0, GroupContains(groups, 3));
}

TEST_CASE("should sort sort fitness 2") {
  constexpr int cryteriaCount = 2;
  thrust::device_vector<FloatArray<cryteriaCount>> fitnesses(5);
  fitnesses[0] = {0, 2};
  fitnesses[1] = {6, 16};
  fitnesses[2] = {8, 11};
  fitnesses[3] = {7, 12};
  fitnesses[4] = {1, 1};

  NonDominatedSorting<cryteriaCount> s(5);
  thrust::device_vector<int> sorted(5);
  auto groups = s.sort(fitnesses, sorted);

  REQUIRE(groups.size() == 3);

  REQUIRE(groups[1] - groups[0] == 3); // size of first group
  REQUIRE(groups[2] - groups[1] == 2); // size of second group

  CHECK_THAT(1, GroupContains(groups, 0));
  CHECK_THAT(2, GroupContains(groups, 0));
  CHECK_THAT(3, GroupContains(groups, 0));

  CHECK_THAT(0, GroupContains(groups, 1));
  CHECK_THAT(4, GroupContains(groups, 1));
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