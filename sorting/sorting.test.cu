#include <thrust/device_vector.h>
#include <sstream>
#include "../catch.h"
#include "../nsga/PopFitness.cuh"
#include "./sorting.cuh"

class GroupContains : public Catch::MatcherBase<int> {
  thrust::host_vector<thrust::device_ptr<int>> groups;
  int group;

 public:
  GroupContains(thrust::host_vector<thrust::device_ptr<int>> groups, int group)
      : groups(groups), group(group) {}

  // Performs the test for this matcher
  virtual bool match(int const& nidle) const override {
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

template <int cryteriaCount>
class GroupContainsPop : public Catch::MatcherBase<int> {
  thrust::host_vector<thrust::device_ptr<PopFitness<cryteriaCount>>> groups;
  int group;

 public:
  GroupContainsPop(
      thrust::host_vector<thrust::device_ptr<PopFitness<cryteriaCount>>> groups,
      int group)
      : groups(groups), group(group) {}

  // Performs the test for this matcher
  virtual bool match(int const& nidle) const override {
    auto begin = groups[group];
    auto end = groups[group + 1];
    for (int i = 0; i < end - begin; i++) {
      PopFitness<cryteriaCount> p = begin[i];
      if (p.index == nidle) {
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
  thrust::host_vector<thrust::device_ptr<int>> groups = s.sort(fitnesses);

  REQUIRE(groups.size() == 5);

  REQUIRE(groups[1] - groups[0] == 1);  // size of first group
  REQUIRE(groups[2] - groups[1] == 2);  // size of second group
  REQUIRE(groups[3] - groups[2] == 1);  // size of third group
  REQUIRE(groups[4] - groups[3] == 1);  // size of forth group

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
  auto groups = s.sort(fitnesses);

  REQUIRE(groups.size() == 3);

  REQUIRE(groups[1] - groups[0] == 3);  // size of first group
  REQUIRE(groups[2] - groups[1] == 2);  // size of second group

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

TEST_CASE("sortHalf") {
  constexpr int cryteriaCount = 2;
  thrust::device_vector<FloatArray<cryteriaCount>> fitnesses(5);
  fitnesses[0] = {0, 2};
  fitnesses[1] = {6, 16};
  fitnesses[2] = {8, 11};
  fitnesses[3] = {7, 12};
  fitnesses[4] = {1, 1};

  NonDominatedSorting<cryteriaCount> s(5);
  auto groups = s.sortHalf(fitnesses);

  REQUIRE(groups.size() == 2);

  REQUIRE(groups[1] - groups[0] == 3);  // size of first group

  CHECK_THAT(1, GroupContains(groups, 0));
  CHECK_THAT(2, GroupContains(groups, 0));
  CHECK_THAT(3, GroupContains(groups, 0));
}

TEST_CASE("sortHalfPop") {
  constexpr int cryteriaCount = 2;
  thrust::device_vector<PopFitness<cryteriaCount>> fitnesses(5);
  fitnesses[0] = {0, nullptr, {0, 2}};
  fitnesses[1] = {1, nullptr, {100, 100}};
  fitnesses[2] = {2, nullptr, {8, 11}};
  fitnesses[3] = {3, nullptr, {7, 12}};
  fitnesses[4] = {4, nullptr, {1, 1}};

  NonDominatedSorting<cryteriaCount> s(5);
  auto groups = s.sortHalfPop(fitnesses);

  // REQUIRE(groups.size() == 3);

  REQUIRE(groups[1] - groups[0] == 1);  // size of first group
  REQUIRE(groups[2] - groups[1] == 2);  // size of second group

  CHECK_THAT(1, GroupContainsPop<cryteriaCount>(groups, 0));
  CHECK_THAT(2, GroupContainsPop<cryteriaCount>(groups, 1));
  CHECK_THAT(3, GroupContainsPop<cryteriaCount>(groups, 1));
}

TEST_CASE("sortHalfPop 2") {
  constexpr int cryteriaCount = 2;
  thrust::device_vector<PopFitness<cryteriaCount>> fitnesses(5);
  fitnesses[0] = {0, nullptr, {0, 0}};
  fitnesses[1] = {1, nullptr, {0, 0}};
  fitnesses[2] = {2, nullptr, {2, 2}};
  fitnesses[3] = {3, nullptr, {3, 3}};
  fitnesses[4] = {4, nullptr, {4, 4}};

  NonDominatedSorting<cryteriaCount> s(5);
  auto groups = s.sortHalfPop(fitnesses);

  REQUIRE(groups.size() == 5);

  // REQUIRE(groups[1] - groups[0] == 3);  // size of first group

  // CHECK_THAT(1, GroupContainsPop<cryteriaCount>(groups, 0));
  // CHECK_THAT(2, GroupContainsPop<cryteriaCount>(groups, 0));
  // CHECK_THAT(3, GroupContainsPop<cryteriaCount>(groups, 0));
}