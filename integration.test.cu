

#include <thrust/host_vector.h>
#include <iostream>
#include "./catch.h"

// #include "./genetics/GeneticAlgorithm.cuh"

#include "./genetics/Population.cuh"

#include "./Performance/Performance.h"
#include "./files/dataset.h"

#include "./files/dataset.h"
#include "./fitness/BasicFitness.cuh"
#include "./fitness/KnnFitness.cuh"
#include "./nsga/genetics.cuh"

#include <sstream>
#include "./files/IrisFitness.cuh"

#include <iostream>

#include <omp.h>

template <typename FitnessType>
void run(int k, std::string name) {
  #pragma omp parallel for
  for (int popSizeI = 1; popSizeI <= 10; popSizeI += 3) {
    std::cout << popSizeI << " iteration" << std::endl;
    for (int iterationsI = 1; iterationsI <= 10; iterationsI += 4) {
      const int popSize = popSizeI * 40;
      const int iterations = iterationsI * 100;
      FitnessType fitness(popSize);
      Genetics<decltype(fitness), 2> ggg(popSize, fitness.size(), &fitness);

      ggg.run(iterations);
      std::stringstream fileName;
      fileName << "results/" << name << "[popSize_" << popSize
               << "][iterations_" << iterations << "]"
               << "[k_" << k << "].csv";
      std::cout << fileName.str() << std::endl;
      fitness.exportResult(ggg.p1, fileName.str());
    }
  }
};

TEST_CASE("Integration", "[integration]") {
//   run<RingFitness<1>>(1,"ring");
//   run<RingFitness<3>>(3,"ring");
//   run<RingFitness<5>>(5,"ring");
//   run<RingFitness<7>>(7,"ring");
//   run<RingFitness<9>>(9,"ring");

//   run<IrisFitness<1>>(1,"iris");
//   run<IrisFitness<3>>(3, "iris");
//   run<IrisFitness<5>>(5, "iris");
//   run<IrisFitness<7>>(7, "iris");
//   run<IrisFitness<9>>(9,"iris");

//   run<BananaFitness<1>>(1,"banana");
  // run<BananaFitness<3>>(3,"banana");
  // run<BananaFitness<5>>(5,"banana");
  // run<BananaFitness<7>>(7,"banana");
//   run<BananaFitness<9>>(9,"banana");

//   run<MagicFitness<1>>(1,"maginc");
  run<MagicFitness<3>>(3, "magic");
  run<MagicFitness<5>>(5, "magic");
  run<MagicFitness<7>>(7, "magic");
//   run<MagicFitness<9>>(9,"maginc");

//   run<SpamBaseFitness<1>>(1, "spam");
  run<SpamBaseFitness<3>>(3, "spam");
  run<SpamBaseFitness<5>>(5, "spam");
  run<SpamBaseFitness<7>>(7, "spam");
//   run<SpamBaseFitness<9>>(9, "spam");

  //   for (int popSizeI = 1; popSizeI <= 10; popSizeI++) {
  //     for (int iterationsI = 1; iterationsI <= 10; iterationsI++) {
  //       const int popSize = popSizeI * 40;
  //       const int iterations = iterationsI * 100;
  //       RingFitness<k> fitness(popSize);
  //       Genetics<decltype(fitness), 2> ggg(popSize, fitness.size(),
  //       &fitness);

  //       ggg.run(iterations);
  //       std::stringstream fileName;
  //       fileName << "results/iris-[popSize_" << popSize << "][iterations_"
  //                << iterations << "]";
  //       std::cout << fileName.str() << std::endl;
  //       fitness.exportResult(ggg.p1, fileName.str());
  //     }
  //   }

  // ggg.exportResult("results/export.csv");
  // knnFitnes.exportResult(ggg.p1, trainLoader.dataSet, testLoader.dataSet,
  // "results/export.csv");

  // Knn<attributeCount, labelsCount, k> knn(trainLoader.dataSet,
  // testLoader.dataSet); thrust::device_vector<float> f(popSize);

  // for(FloatArray<2> el: f) {

  // }
};

// #include "./catch.h"
// #include <iostream>
// #include <thrust/host_vector.h>

// // #include "./genetics/GeneticAlgorithm.cuh"

// #include "./genetics/Population.cuh"

// #include "./Performance/Performance.h"
// #include "./files/dataset.h"

// #include "./fitness/BasicFitness.cuh"
// #include "./fitness/KnnFitness.cuh"
// #include "./nsga/genetics.cuh"
// #include "./files/dataset.h"

// #include "./files/IrisFitness.cuh"

// TEST_CASE("Integration", "[integration]") {

//     constexpr int k = 9;
//     constexpr int popSize = 300;

//     IrisFitness<k> fitness(popSize);
// z
//     // DataSetLoader<4>
//     trainLoader("./processDataset/data/iris/iris-train.csv");
//     // DataSetLoader<4>
//     testLoader("./processDataset/data/iris/iris-train.csv");
//     // constexpr int attributeCount = 4;
//     // constexpr int labelsCount = 3;
//     // KnnFitnessNSGA<attributeCount, labelsCount, k> knnFitnes(popSize,
//     trainLoader.dataSet); Genetics<decltype(fitness), 2> ggg(popSize,
//     fitness.size(), &fitness); ggg.mutation.rate = 0.002f; ggg.run(150);
//     fitness.exportResult(ggg.p1, "results/iris.csv");
//     // ggg.exportResult("results/export.csv");
//     // knnFitnes.exportResult(ggg.p1, trainLoader.dataSet,
//     testLoader.dataSet, "results/export.csv");

//     // Knn<attributeCount, labelsCount, k> knn(trainLoader.dataSet,
//     testLoader.dataSet);
//     // thrust::device_vector<float> f(popSize);

//     // for(FloatArray<2> el: f) {

//     // }

// };
