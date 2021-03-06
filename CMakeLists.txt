cmake_minimum_required (VERSION 3.0)
enable_language(CUDA)
project (Optimization)
include_directories(./)

set(CMAKE_CUDA_FLAGS "--default-stream per-thread --expt-extended-lambda -arch=sm_50 -Xptxas -O3 -Xcompiler -fopenmp -Xcompiler -O3")
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set( CMAKE_CXX_FLAGS_RELEASE "-O3")

add_executable(Optimization.out
    tests.cpp
    ./nsga/selection.cu
    ./nsga/selection.test.cu
    ./nsga/crossover.test.cu

    ./nsga/genetics.test.cu


    ./crowdingDistance/crowdingDistance.test.cu

    ./fitness/fitnessTransform.cu
    ./fitness/KnnFitness.test.cu
    ./fitness/populationReduction.test.cu

    ./genetics/genetics.test.cu
    ./genetics/Randomize.cu

    ./Knn/Knn.test.cu
    ./Performance/Performance.cu
    ./sorting/sorting.test.cu
    ./integration.test.cu
)

# add_executable(Optimization.a.out
#     ./main.cu
#     ./nsga/selection.cu
#     ./fitness/fitnessTransform.cu
#     ./genetics/Randomize.cu
#     ./Performance/Performance.cpp
# )

find_library(GOMP_LIBRARY gomp)

if (NOT GOMP_LIBRARY)

  message(SEND_ERROR "gomp library not found")

endif()

# target_link_libraries(mytarget )

target_link_libraries (Optimization.out libcurand.so ${GOMP_LIBRARY})
# target_link_libraries (Optimization.a.out libcurand.so)
