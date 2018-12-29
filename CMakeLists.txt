cmake_minimum_required (VERSION 3.0)
enable_language(CUDA)
project (Optimization)
include_directories(./)

set(CMAKE_CUDA_FLAGS "--expt-extended-lambda")


add_executable(Optimization.out
    tests.cpp
    ./nsga/selection.cu
    ./nsga/selection.test.cu

    ./crowdingDistance/crowdingDistance.test.cu

    ./fitness/fitnessTransform.cu
    ./fitness/KnnFitness.test.cu
    ./fitness/populationReduction.test.cu

    ./genetics/genetics.test.cu
    ./genetics/Randomize.cu
    ./Knn/Knn.test.cu
    ./Performance/Performance.cpp
    ./sorting/sorting.test.cu

)