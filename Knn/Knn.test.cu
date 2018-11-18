
#include "../files/dataset.h"
#include "../fitness/populationReduction.cuh"
#include "./Knn.cuh"
#include <thrust/device_vector.h>

#include "../catch.h"

TEST_CASE("Knn should calculate correct accuracy", "[Knn]") {
    DataSetLoader<4> loader("./Knn/testData1.csv");
    Knn<4, 2, 3> knn(loader.dataSet);
    thrust::device_vector<bool> testPop(loader.dataSet.size(), true);
    Population<bool> p(1, static_cast<int>(loader.dataSet.size()));
    thrust::device_vector<float> accuracy(1);
    knn.accuracy(p, accuracy);

    REQUIRE(accuracy[0] == Approx(6.0));
}

TEST_CASE("Knn accuracy should take magority of voters as label", "[Knn]") {
    DataSetLoader<4> loader("./Knn/testData2.csv");
    Knn<4, 4, 5> knn(loader.dataSet);
    Population<bool> p(1, static_cast<int>(loader.dataSet.size()));

    thrust::device_vector<float> accuracy(1);
    knn.accuracy(p, accuracy);
    REQUIRE(accuracy[0] == Approx(3.0f));
}

TEST_CASE("Knn should exclude vectors not preset in genom", "[Knn]") {
    DataSetLoader<4> loader("./Knn/testData1.csv");
    Knn<4, 2, 3> knn(loader.dataSet);
    Population<bool> p(1, static_cast<int>(loader.dataSet.size()));
    p.population[0] = false;
    p.population[1] = false;
    thrust::device_vector<float> accuracy(1);
    knn.accuracy(p, accuracy);

    REQUIRE(accuracy[0] == Approx(3.0f));
}

TEST_CASE("Knn should calculate correct accuracy betwean training and test",
          "[Knn]") {
    DataSetLoader<4> trainLoader("./Knn/train.csv");
    DataSetLoader<4> testLoader("./Knn/testTest.csv");

    Knn<4, 4, 3> knn(trainLoader.dataSet, testLoader.dataSet);
    Population<bool> p(1, static_cast<int>(trainLoader.dataSet.size()));
    thrust::device_vector<float> accuracy(1);
    knn.accuracy(p, accuracy);

    REQUIRE(accuracy[0] == Approx(2.0f));
}

TEST_CASE("Knn should calculate correct accuracy betwean training and test when some gens are mising",
          "[Knn]") {
    DataSetLoader<4> trainLoader("./Knn/train.csv");
    DataSetLoader<4> testLoader("./Knn/testTest.csv");

    Knn<4, 4, 3> knn(trainLoader.dataSet, testLoader.dataSet);
    Population<bool> p(1, static_cast<int>(testLoader.dataSet.size()));
    p.population[0] = false;
    thrust::device_vector<float> accuracy(1);
    knn.accuracy(p, accuracy);

    REQUIRE(accuracy[0] == Approx(1.0f));
}
