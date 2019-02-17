#pragma once
#include "../lib/FloatArray.cuh"

template<int cryteriaCount>
struct PopFitness {
  int index;
  bool* specimen;
  FloatArray<cryteriaCount> fitness;
};
