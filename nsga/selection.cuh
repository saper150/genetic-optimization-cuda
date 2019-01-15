#pragma once
#include <thrust/device_vector.h>

void fillFronts(const thrust::host_vector<thrust::device_ptr<int>>& groups,
                thrust::device_vector<int>& fronts);

void fillFrontsActual(const thrust::host_vector<thrust::device_ptr<int>>& groups,
                thrust::device_vector<int>& fronts);


// const int size = groups.back() - groups.front();
// thrust::transform(
//     thrust::make_counting_iterator<int>(0),
//     thrust::make_counting_iterator<int>(size),
//     fronts.begin(),[]__device__() {

//     },
// )


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
