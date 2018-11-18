#pragma once

#include "../ariaParser.h"
#include <fstream>
#include <string>
#include <thrust/host_vector.h>
#include <unordered_map>

template <typename T, int size> struct DataPoint {
    float position[size];
    T label;
};

template <int size> using LabelDataPoint = DataPoint<int, size>;
// template <int size> using ClasificationDataPoint = DataPoint<int, size>;

template <int size> struct DataSetLoader {

  public:
    thrust::host_vector<LabelDataPoint<size>> dataSet;

    DataSetLoader(std::string fileName) {
        std::ifstream fileStream(fileName);
        aria::csv::CsvParser parser(fileStream);

        for (const auto &row : parser) {
            DataPoint<int, size> point;

            for (size_t i = 0; i < size; i++) {
                point.position[i] = std::stod(row[i]);
            }
            // const std::string label = row[size];
            point.label = std::stoi(row[size]);
            dataSet.push_back(point);
        }
    }
};

template <int size>
thrust::host_vector<DataPoint<int, size>> loadDataset(std::string fileName) {

    return thrust::host_vector<DataPoint<int, size>>();
}
