
#include <iostream>
#include <thrust/host_vector.h>

#include "./genetics/GeneticAlgorithm.cuh"

#include "./genetics/Population.cuh"

#include "./Performance/Performance.h"
#include "./files/dataset.h"

#include "./fitness/BasicFitness.cuh"
#include "./fitness/KnnFitness.cuh"

int main() {

    try {
        std::cout << "start" << std::endl;
        DataSetLoader<4> loader("./processDataset/data/iris/iris-verify.csv");

        KnnFitness<4> knnFitness(loader.dataSet);

        BasicFitness<bool> basicFitness(100);

        Performance::mesure("all", [&]() {
            GeneticAlgorithm<bool, KnnFitness<4>> gen(200, knnFitness);

            for (int i = 0; i < 1000; i++) {
                gen.iterate();
            }
        });
        Performance::print();
    } catch (const std::runtime_error &re) {
        std::cout << "Runtime error: " << re.what() << std::endl;
    } catch (const std::exception &ex) {
        std::cout << "Error occurred: " << ex.what() << std::endl;
    } catch (...) {
        std::cout << "Unknown failure occurred. Possible memory corruption"
                  << std::endl;
    }
}