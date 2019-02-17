#pragma once
#include <thrust/device_vector.h>
#include "./PopFitness.cuh"

inline void fillFronts(
    const thrust::host_vector<thrust::device_ptr<int>>& groups,
    thrust::device_vector<int>& fronts) {
  for (int i = 1; i < groups.size(); i++) {
    thrust::for_each(groups[i - 1], groups[i],
                     [front = i - 1, fronts = thrust::raw_pointer_cast(
                                         fronts.data())] __device__(int el) {
                       fronts[el] = front;
                     });
  }
}

template <int cryteriaCount>
inline void fillFrontsPop(
    const thrust::host_vector<thrust::device_ptr<PopFitness<cryteriaCount>>>&
        groups,
    thrust::device_vector<int>& fronts) {
  for (int i = 1; i < groups.size(); i++) {
    thrust::for_each(
        groups[i - 1], groups[i],
        [front = i - 1,
         fronts = thrust::raw_pointer_cast(
             fronts.data())] __device__(PopFitness<cryteriaCount> el) {
          fronts[el.index] = front;
        });
  }
}

inline __device__ int winner(const int* fronts,
                             const float* distances,
                             int a,
                             int b) {
  if (fronts[a] == fronts[b]) {
    return distances[a] > distances[b] ? a : b;
  } else {
    return fronts[a] > fronts[b] ? b : a;
  }
};

template<int cryteriaCount>
__device__ bool* winnerPop(const int* fronts,
                             const float* distances,
                             PopFitness<cryteriaCount> a,
                             PopFitness<cryteriaCount> b) {
  if (fronts[a.index] == fronts[b.index]) {
    return distances[a.index] > distances[b.index] ? a.specimen : b.specimen;
  } else {
    return fronts[a.index] > fronts[b.index] ? b.specimen : a.specimen;
  }
};


template <int cryteriaCount>
struct Selection {
  int popSize;
  thrust::device_vector<thrust::tuple<int, int>> pairs;
  thrust::device_vector<thrust::tuple<bool*, bool*>> pairsPop;
  thrust::device_vector<int> fronts;

  Selection(int popSize) : popSize(popSize), pairs(popSize), fronts(popSize), pairsPop(popSize) {}

  void select(const thrust::host_vector<thrust::device_ptr<int>>& groups,
              const thrust::device_vector<float>& distances,
              const thrust::device_vector<int>& rng) {
    fillFronts(groups, fronts);

    thrust::for_each_n(
        thrust::make_counting_iterator<int>(0), popSize,
        [g = groups[0].get(), rng = thrust::raw_pointer_cast(rng.data()),
         fronts = thrust::raw_pointer_cast(fronts.data()),
         distances = thrust::raw_pointer_cast(distances.data()),
         pairs = thrust::raw_pointer_cast(pairs.data())] __device__(int i) {
          const int p1 = i * 4;
          const int p2 = i * 4 + 1;
          const int p3 = i * 4 + 2;
          const int p4 = i * 4 + 3;

          pairs[i] = thrust::make_tuple<int, int>(
              winner(fronts, distances, g[rng[p1]], g[rng[p2]]),
              winner(fronts, distances, g[rng[p3]], g[rng[p4]]));
        });
  }
  void selectPop(const thrust::host_vector<
                     thrust::device_ptr<PopFitness<cryteriaCount>>>& groups,
                 const thrust::device_vector<float>& distances,
                 const thrust::device_vector<int>& rng) {
    fillFrontsPop(groups, fronts);
    thrust::for_each_n(
        thrust::make_counting_iterator<int>(0), popSize,
        [g = groups[0].get(), rng = thrust::raw_pointer_cast(rng.data()),
         fronts = thrust::raw_pointer_cast(fronts.data()),
         distances = thrust::raw_pointer_cast(distances.data()),
         pairs = thrust::raw_pointer_cast(pairsPop.data())] __device__(int i) {
          const int p1 = i * 4;
          const int p2 = i * 4 + 1;
          const int p3 = i * 4 + 2;
          const int p4 = i * 4 + 3;

          pairs[i] = thrust::make_tuple<bool*, bool*>(
              winnerPop(fronts, distances, g[rng[p1]], g[rng[p2]]),
              winnerPop(fronts, distances, g[rng[p3]], g[rng[p4]]));
        });
  }
};

// void selection(const thrust::host_vector<thrust::device_ptr<int>>& groups,
//                const thrust::device_vector<float>& distances,
//                const thrust::device_vector<int>& rng,
//                thrust::device_vector<thrust::tuple<int, int>>& outputPairs)
//                {

//                }
