#include "./selection.cuh"

void fillFronts(const thrust::host_vector<thrust::device_ptr<int>>& groups,
                thrust::device_vector<int>& fronts) {
  for (int i = 1; i < groups.size(); i++) {
    thrust::for_each(groups[i - 1], groups[i],
                     [front = i - 1, fronts = thrust::raw_pointer_cast(
                                         fronts.data())] __device__(int el) {
                       fronts[el] = front;
                     });
  }
}


__device__ int winner(const int* fronts, const float* distances, int a, int b) {
  if (fronts[a] == fronts[b]) {
    return distances[a] > distances[b] ? a : b;
  } else {
    return fronts[a] > fronts[b] ? b : a;
  }
};

Selection::Selection(int popSize)
    : popSize(popSize), pairs(popSize), fronts(popSize){};

void Selection::select(
    const thrust::host_vector<thrust::device_ptr<int>>& groups,
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
