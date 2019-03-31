


#include "./catch.h"
#include <iostream>
#include <thrust/host_vector.h>

// #include "./genetics/GeneticAlgorithm.cuh"

#include "./genetics/Population.cuh"

#include "./Performance/Performance.h"
#include "./files/dataset.h"

#include "./fitness/BasicFitness.cuh"
#include "./fitness/KnnFitness.cuh"
#include "./nsga/genetics.cuh"

#include "./files/dataset.h"

TEST_CASE("Integration", "[integration]") {
    constexpr int popSize = 100;
    DataSetLoader<4> trainLoader("./processDataset/data/iris/iris-train.csv");
    DataSetLoader<4> testLoader("./processDataset/data/iris/iris-train.csv");
    constexpr int attributeCount = 4;
    constexpr int labelsCount = 3;
    constexpr int k = 9;
    KnnFitnessNSGA<attributeCount, labelsCount, k> knnFitnes(popSize, trainLoader.dataSet);

    Genetics<decltype(knnFitnes), 2> ggg(popSize, trainLoader.dataSet.size(), &knnFitnes);
    // ggg.exportResult("results/export.csv");
    knnFitnes.exportResult(ggg.p1, trainLoader.dataSet, testLoader.dataSet, "results/export.csv");

    // Knn<attributeCount, labelsCount, k> knn(trainLoader.dataSet, testLoader.dataSet);
    // thrust::device_vector<float> f(popSize);

    // for(FloatArray<2> el: f) {

    // }


};
