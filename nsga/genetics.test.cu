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
  thrust::device_vector<FloatArray<2>> operator()(Population<bool>& p) {
    // DevicePopulation<bool> devicePop = p.toDevicePopulation();

    thrust::device_vector<FloatArray<2>> result (p.popSize());

    for (int i = 0; i < p.popSize(); i++) {
      FloatArray<2> f;
        f.data[0] = 0;
        f.data[1] = 0;
        for(int j = 0; j < p.genSize; j++) {
            f.data[0]+= p.population[i * p.genSize + j];
        }
        result[i] = f;
    }

    return result;
  }
};

TEST_CASE("genetics") {
  constexpr int popSize = 100;
  constexpr int genSize = 4;
  // thrust::tuple<>
  std::function<FloatArray<2>(Population<bool> & p)> f = [](auto& p) {
    return FloatArray<2>();
  };

  Genetics<TestFitness, 2> ggg(popSize, genSize, TestFitness());

  REQUIRE(true == false);
}
