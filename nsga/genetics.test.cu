#include <thrust/tuple.h>
#include "../catch.h"
#include "../lib/FloatArray.cuh"
#include "./genetics.cuh"

#include "../Knn/Knn.cuh"
#include "../files/dataset.h"
#include "../fitness/populationReduction.cuh"

#include <thrust/device_vector.h>
#include <thrust/reduce.h>

// template<typename T>
// struct positive_value : public thrust::unary_function<T,T>
// {
//    __host__ __device__ T operator()(const T &x) const
//    {
//      return x < T(0) ? 0  : x;
//    }
// };

// float result = thrust::transform_reduce(data.begin(), data.end(),
//                                     positive_value<float>(),
//                                     0,
//                                     thrust::plus<float>());

struct TestFitness {
  void operator()(Population<bool>& p,
                  thrust::device_vector<FloatArray<2>>& dest) {
    thrust::host_vector<bool> hostPop = p.population;
    thrust::device_vector<FloatArray<2>> f(p.popSize());
    for (int i = 0; i < p.popSize(); i++) {
      FloatArray<2> f;
      f.data[0] = 0;
      f.data[1] = 0;
      for (int j = 0; j < p.genSize; j++) {
        if (j % 2 == 0 && hostPop[i * p.genSize + j]) {
          f.data[0] += 1;
        }

        if (j % 2 != 0 && !hostPop[i * p.genSize + j]) {
          f.data[0] += 1;
        }

        f.data[1] += hostPop[i * p.genSize + j] ? 1 : 0;
      }
      dest[i] = f;
    }
  }
};

TEST_CASE("genetics") {
  constexpr int popSize = 100;
  constexpr int genSize = 20;

  auto f = TestFitness();
  Genetics<TestFitness, 2> ggg(popSize, genSize, &f);

  REQUIRE(true == false);
};

TEST_CASE("genetics knn") {
  DataSetLoader<4> loader("./Knn/testData1.csv");
  Knn<4, 2, 3> knn(loader.dataSet);
  thrust::device_vector<bool> testPop(loader.dataSet.size(), true);
  Population<bool> p(1, static_cast<int>(loader.dataSet.size()));
  thrust::device_vector<float> accuracy(1);
  knn.accuracy(p, accuracy);

  constexpr int popSize = 100;
  constexpr int genSize = 20;

  auto f = TestFitness();
  Genetics<TestFitness, 2> ggg(popSize, genSize, &f);

  REQUIRE(true == false);
};

TEST_CASE("sortLastGroup") {
  thrust::device_vector<int> groupsVector(5);
  groupsVector[0] = 0;
  groupsVector[1] = 1;
  groupsVector[2] = 2;
  groupsVector[3] = 3;
  groupsVector[4] = 4;

  thrust::host_vector<thrust::device_ptr<int>> groups(3);
  groups[0] = thrust::device_pointer_cast(groupsVector.data());
  groups[1] = thrust::device_pointer_cast(groupsVector.data() + 2);
  groups[2] = thrust::device_pointer_cast(groupsVector.data() + 5);

  thrust::device_vector<float> distances(5);
  distances[2] = 0;
  distances[3] = 100;
  distances[4] = 50;

  sortLastGroup(groups, distances);

  REQUIRE(groupsVector[0] == 0);
  REQUIRE(groupsVector[1] == 1);
  REQUIRE(groupsVector[2] == 3);
  REQUIRE(groupsVector[3] == 4);
  REQUIRE(groupsVector[4] == 2);
}
