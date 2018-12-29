#pragma once
#include <thrust/device_vector.h>

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

  // const int size = groups.back() - groups.front();
  // thrust::transform(
  //     thrust::make_counting_iterator<int>(0),
  //     thrust::make_counting_iterator<int>(size),
  //     fronts.begin(),[]__device__() {

  //     },
  // )
}

struct Selection {
  int popSize;
  thrust::device_vector<thrust::tuple<int, int>> pairs;
  thrust::device_vector<int> fronts;
  Selection(int popSize);

  void select(const thrust::host_vector<thrust::device_ptr<int>>& groups,
              const thrust::device_vector<float>& distances,
              const thrust::device_vector<int>& rng);
};

// void selection(const thrust::host_vector<thrust::device_ptr<int>>& groups,
//                const thrust::device_vector<float>& distances,
//                const thrust::device_vector<int>& rng,
//                thrust::device_vector<thrust::tuple<int, int>>& outputPairs) {

//                }
