#include "../catch.h"
#include "../files/dataset.h"
#include "./KnnFitness.cuh"

TEST_CASE("knnAlphaFitness should normalize knn accuracy",
          "[knnAlphaFitness]") {

    DataSetLoader<4> loader("./Knn/testData1.csv");
    KnnFitness<4, 2, 3> knn(loader.dataSet);
    Population<bool> p(1, static_cast<int>(loader.dataSet.size()));
    thrust::device_vector<float> fitness(1);
    knn(p, fitness);
    REQUIRE(fitness[0] == Approx(1.0));
}

TEST_CASE("knnAlphaFitness should rise result to given power",
          "[knnAlphaFitness]") {
    DataSetLoader<4> loader("./Knn/testData2.csv");
    KnnFitness<4, 4, 3> knn(loader.dataSet);
    knn.power = 2;
    Population<bool> p(1, static_cast<int>(loader.dataSet.size()));
    thrust::device_vector<float> fitness(1);
    knn(p, fitness);

    REQUIRE(fitness[0] == Approx(0.5f * 0.5f));
}

TEST_CASE("knnAlphaFitness take genSize into account", "[knnAlphaFitness]") {

    DataSetLoader<4> loader("./Knn/testData1.csv");
    KnnFitness<4, 2, 3> knn(loader.dataSet);
    knn.power = 2;
    Population<bool> p(1, static_cast<int>(loader.dataSet.size()));
    p.population[0] = false;
    p.population[1] = false;
    thrust::device_vector<float> fitness(1);
    knn(p, fitness);

    REQUIRE(fitness[0] == Approx(powf(0.5f + 1.f - 4.f / 6.f, 2)));
}

TEST_CASE("knnAlphaFitness should set fitness to 0 if number of speciments is lower than k", "[knnAlphaFitness]") {

    DataSetLoader<4> loader("./Knn/testData1.csv");
    KnnFitness<4, 2, 3> knn(loader.dataSet);
    Population<bool> p(1, static_cast<int>(loader.dataSet.size()));
    p.population[5] = false;
    p.population[4] = false;
    p.population[3] = false;
    p.population[2] = false;
    thrust::device_vector<float> fitness(1);
    knn(p, fitness);
    REQUIRE(fitness[0] == 0.f);
}
