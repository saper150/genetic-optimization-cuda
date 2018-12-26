#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/transform.h>

inline void fillRandom(thrust::device_vector<float>& randomNumbers) {
  thrust::transform(thrust::make_counting_iterator<int>(0),
                    thrust::make_counting_iterator<int>(randomNumbers.size()),
                    randomNumbers.begin(), randomNumbers.begin(),
                    [] __device__(int index) {
                      thrust::default_random_engine rng;
                      thrust::uniform_real_distribution<float> dist(0.f, 1.f);
                      rng.discard(index);
                      return dist(rng);
                    });
}

