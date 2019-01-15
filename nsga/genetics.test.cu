#include <thrust/tuple.h>
#include "../catch.h"
#include "../lib/FloatArray.cuh"
#include "./genetics.cuh"

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
  thrust::device_vector<FloatArray<2>> operator()(
      Population<bool>& p,
      thrust::device_ptr<FloatArray<2>> dest) {
    // DevicePopulation<bool> devicePop = p.toDevicePopulation();

    thrust::device_vector<FloatArray<2>> result(p.popSize());

    for (int i = 0; i < p.popSize(); i++) {
      FloatArray<2> f;
      f.data[0] = 0;
      f.data[1] = 0;
      for (int j = 0; j < p.genSize; j++) {
        f.data[0] += p.population[i * p.genSize + j];
      }
      *(dest + i) = f;
      // result[i] = f;
    }

    return result;
  }
};

TEST_CASE("genetics") {
  constexpr int popSize = 10;
  constexpr int genSize = 4;
  // thrust::tuple<>
  std::function<FloatArray<2>(Population<bool> & p)> f = [](auto& p) {
    return FloatArray<2>();
  };

  Genetics<TestFitness, 2> ggg(popSize, genSize, TestFitness());

  REQUIRE(true == false);
}


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

