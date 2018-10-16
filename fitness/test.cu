#include "../files/dataset.h"
#include "./KnnFItness.cuh"
#include "./test.h"
#include <iostream>
#include <thrust/device_vector.h>

#include "./populationReduction.cuh"

bool compf(float A, float B, float epsilon = 0.005f) {
    return (fabs(A - B) < epsilon);
}

void shouldCalculateCorrectAccuracy() {
    DataSetLoader<4> loader("./fitness/testData1.csv");
    KnnFitness<4, 3> knnFitness(loader.dataSet);
    thrust::device_vector<bool> testPop(loader.dataSet.size(), true);
    Population<bool> p = {testPop, (int)loader.dataSet.size()};

    thrust::device_vector<float> accuracy(1);
    knnFitness.accuracy(p, accuracy);

    if (!compf(accuracy[0], 1.0)) {
        std::cout << "shouldCalculateCorrectAccuracy FAILED" << std::endl;
    }
}

void shouldExcludeVectorsNotpresentInGeneom() {
    DataSetLoader<4> loader("./fitness/testData1.csv");
    KnnFitness<4, 3> knnFitness(loader.dataSet);
    thrust::device_vector<bool> testPop(loader.dataSet.size(), true);
    testPop[0] = false;
    testPop[1] = false;
    Population<bool> p = {testPop, (int)loader.dataSet.size()};

    thrust::device_vector<float> accuracy(1);
    knnFitness.accuracy(p, accuracy);

    std::cout << accuracy[0] << std::endl;
    if (!compf(accuracy[0], .5f)) {
        std::cout << "shouldExcludeVectorsNotpresentInGeneom FAILED"
                  << std::endl;
        std::cout << "expected 0.5f but got " << accuracy[0] << std::endl;
    }
}

void reductionLevelShouldCalculateCorrectReductionLevel() {
    Population<bool> p = {thrust::device_vector<bool>(10, true), 5};
    thrust::device_vector<float> reductionsLevels(p.popSize(), 0.f);
    getReductionLevels<bool>(p, reductionsLevels);

    if (!compf(reductionsLevels[0], 0.f)) {
        std::cout << "reductionLevelShouldCalculateCorrectReductionLevel FAILED"
                  << std::endl;
        std::cout << "expected [0] 0.f got " << reductionsLevels[0] << std::endl;
    }
    if (!compf(reductionsLevels[1], 0.f)) {
        std::cout << "reductionLevelShouldCalculateCorrectReductionLevel FAILED"
                  << std::endl;
        std::cout << "expected [1] 0.f got " << reductionsLevels[0] << std::endl;
    }
}

void reductionLevelShouldCalculateCorrectReductionLevel2() {
    Population<bool> p = {thrust::device_vector<bool>(10, true), 5};
    p.population[0] = false;
    p.population[1] = false;
    p.population[6] = false;

    thrust::device_vector<float> reductionsLevels(p.popSize(), 0.f);
    getReductionLevels<bool>(p, reductionsLevels);

    if (!compf(reductionsLevels[0], 0.4f)) {
        std::cout
            << "reductionLevelShouldCalculateCorrectReductionLevel2 FAILED"
            << std::endl;
        std::cout << "expected 0.4f got " << reductionsLevels[0] << std::endl;
    }

    if (!compf(reductionsLevels[1], 0.2f)) {
        std::cout
            << "reductionLevelShouldCalculateCorrectReductionLevel2 FAILED"
            << std::endl;
        std::cout << "expected 0.2f got " << reductionsLevels[0] << std::endl;
    }
}

void reductionLevelShouldResetReductionLevelValues() {
    Population<bool> p = {thrust::device_vector<bool>(1, true), 1};
    thrust::device_vector<float> reductionsLevels(p.popSize(), 10.f);
    getReductionLevels<bool>(p, reductionsLevels);
    if (!compf(reductionsLevels[0], 0.0f)) {
        std::cout << "reductionLevelShouldResetReductionLevelValues FAILED"
                  << std::endl;
        std::cout << "memory is not reset" << std::endl;
    }
}

void KnnFitnessTest() {
    std::cout << "Knn test run" << std::endl;
    shouldCalculateCorrectAccuracy();
    shouldExcludeVectorsNotpresentInGeneom();
    reductionLevelShouldCalculateCorrectReductionLevel();
    reductionLevelShouldCalculateCorrectReductionLevel2();
    reductionLevelShouldResetReductionLevelValues();
}
