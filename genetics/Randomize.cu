#include <curand.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/random/uniform_int_distribution.h>
#include <cmath>
#include "./Randomize.cuh"

__global__ void randomize(bool* f, const int size) {
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx > size) {
    return;
  }
  curandState state;
  curand_init(clock64(), idx, 0, &state);
  f[idx] = curand_uniform(&state) > .5f ? true : false;
}

void randomize(thrust::device_vector<bool>& vec) {
  const int threadsPerBlock = 200;
  const int blocks = std::ceil((float)vec.size() / threadsPerBlock);
  randomize<<<blocks, threadsPerBlock>>>(thrust::raw_pointer_cast(&vec[0]),
                                         vec.size());
}

// struct GenRand {
//   int max;
//   GenRand(int max) : max(max) {}
//   __device__ float operator()(int idx) {
//     thrust::default_random_engine randEng;
//     thrust::uniform_int_distribution<int> uniDist(0, max);
//     randEng.discard(idx);
//     return uniDist(randEng);
//   }
// };

// void randomize(thrust::device_vector<int>& vec, int max) {

//   curandGenerator_t g;
//   curandCreateGenerator(&g, CURAND_RNG_PSEUDO_DEFAULT);
//   curandSetPseudoRandomGeneratorSeed(g, time(0));

//   curandGenerateUniform(g,

//   thrust::transform(thrust::make_counting_iterator(0),
//                     thrust::make_counting_iterator(static_cast<int>(vec.size())),
//                     vec.begin(), GenRand(max));
// }

IntRng::IntRng(int size, int max)
    : intermidiate(size), res(size), max(max) {
  curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
};

IntRng::~IntRng() {
  curandDestroyGenerator(generator);
}

thrust::device_vector<int>& IntRng::generate() {
  curandSetPseudoRandomGeneratorSeed(generator, time(0));
  curandGenerateUniform(generator,
                        thrust::raw_pointer_cast(intermidiate.data()),
                        intermidiate.size());
  thrust::transform(intermidiate.begin(), intermidiate.end(), res.begin(),
                    [max = max] __device__(float f) { return floorf(f * max); });
  return this->res;
}

// struct SelectionRng {
//   int
//     thrust::device_vector<float> intermidiate;
//     thrust::device_vector<int> res;
//     SelectionRng(int size, int max);

// }