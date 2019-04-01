#pragma once
#include <thrust/device_vector.h>
#include <memory>
#include "../fitness/KnnFitness.cuh"
#include "../lib/FloatArray.cuh"
#include "./dataset.h"

template <int k>
struct IrisFitness {
  DataSetLoader<4> trainLoader;
  DataSetLoader<4> testLoader;

  KnnFitnessNSGA<4, 3, k> fitnessFunc;

  IrisFitness(int popSize)
      : trainLoader("./processDataset/data/iris/iris-train.csv"),
        testLoader("./processDataset/data/iris/iris-test.csv"),
        fitnessFunc(popSize, trainLoader.dataSet) {}

  void operator()(Population<bool>& population,
                  thrust::device_vector<FloatArray<2>>& fitness) {
    fitnessFunc(population, fitness);
  }

  int size() {
      return trainLoader.dataSet.size();
  }

  void exportResult(Population<bool>& population, std::string fileName) {
    fitnessFunc.exportResult(population, trainLoader.dataSet, testLoader.dataSet,
                           fileName);
  }
};


template <int k, int attributeCount = 2>
struct BananaFitness {
  DataSetLoader<attributeCount> trainLoader;
  DataSetLoader<attributeCount> testLoader;

  KnnFitnessNSGA<attributeCount, 2, k> fitnessFunc;

  BananaFitness(int popSize)
      : trainLoader("./processDataset/data/banana/banana-train.csv"),
        testLoader("./processDataset/data/banana/banana-train.csv"),
        fitnessFunc(popSize, trainLoader.dataSet) {}

  void operator()(Population<bool>& population,
                  thrust::device_vector<FloatArray<2>>& fitness) {
    fitnessFunc(population, fitness);
  }

  int size() {
      return trainLoader.dataSet.size();
  }

  void exportResult(Population<bool>& population, std::string fileName) {
    fitnessFunc.exportResult(population, trainLoader.dataSet, testLoader.dataSet,
                           fileName);
  }
};


template <int k, int attributeCount = 10>
struct MagicFitness {
  DataSetLoader<attributeCount> trainLoader;
  DataSetLoader<attributeCount> testLoader;

  KnnFitnessNSGA<attributeCount, 2, k> fitnessFunc;

  MagicFitness(int popSize)
      : trainLoader("./processDataset/data/magic/magic-train.csv"),
        testLoader("./processDataset/data/magic/magic-test.csv"),
        fitnessFunc(popSize, trainLoader.dataSet) {}

  void operator()(Population<bool>& population,
                  thrust::device_vector<FloatArray<2>>& fitness) {
    fitnessFunc(population, fitness);
  }

  int size() {
      return trainLoader.dataSet.size();
  }

  void exportResult(Population<bool>& population, std::string fileName) {
    fitnessFunc.exportResult(population, trainLoader.dataSet, testLoader.dataSet,
                           fileName);
  }
};

template <int k, int attributeCount = 20>
struct RingFitness {
  DataSetLoader<attributeCount> trainLoader;
  DataSetLoader<attributeCount> testLoader;

  KnnFitnessNSGA<attributeCount, 2, k> fitnessFunc;

  RingFitness(int popSize)
      : trainLoader("./processDataset/data/ring/ring-train.csv"),
        testLoader("./processDataset/data/ring/ring-test.csv"),
        fitnessFunc(popSize, trainLoader.dataSet) {}

  void operator()(Population<bool>& population,
                  thrust::device_vector<FloatArray<2>>& fitness) {
    fitnessFunc(population, fitness);
  }

  int size() {
      return trainLoader.dataSet.size();
  }

  void exportResult(Population<bool>& population, std::string fileName) {
    fitnessFunc.exportResult(population, trainLoader.dataSet, testLoader.dataSet,
                           fileName);
  }
};


template <int k, int attributeCount = 57>
struct SpamBaseFitness {
  DataSetLoader<attributeCount> trainLoader;
  DataSetLoader<attributeCount> testLoader;

  KnnFitnessNSGA<attributeCount, 2, k> fitnessFunc;

  SpamBaseFitness(int popSize)
      : trainLoader("./processDataset/data/spambase/spambase-train.csv"),
        testLoader("./processDataset/data/spambase/spambase-test.csv"),
        fitnessFunc(popSize, trainLoader.dataSet) {}

  void operator()(Population<bool>& population,
                  thrust::device_vector<FloatArray<2>>& fitness) {
    fitnessFunc(population, fitness);
  }

  int size() {
      return trainLoader.dataSet.size();
  }

  void exportResult(Population<bool>& population, std::string fileName) {
    fitnessFunc.exportResult(population, trainLoader.dataSet, testLoader.dataSet,
                           fileName);
  }
};
