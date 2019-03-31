
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

int main() {

    constexpr int popSize = 100;

    DataSetLoader<4> loader("./processDataset/data/iris/iris-train.csv");
    constexpr int attributeCount = 4;
    constexpr int labelsCount = 3;
    constexpr int k = 3;
    KnnFitnessNSGA<attributeCount, labelsCount, k> knn(popSize, loader.dataSet);

    Genetics<decltype(knn), 2> ggg(popSize, loader.dataSet.size(), &knn);

    try {
        cudaDeviceSynchronize();
        Performance::print();
    } catch (const std::runtime_error &re) {
        std::cout << "Runtime error: " << re.what() << std::endl;
    } catch (const std::exception &ex) {
        std::cout << "Error occurred: " << ex.what() << std::endl;
    } catch (const char *message) {
        std::cout << "ERROR: " << message << std::endl;
    } catch (...) {
        std::cout << "Unknown failure occurred. Possible memory corruption"
                  << std::endl;
    }
}

