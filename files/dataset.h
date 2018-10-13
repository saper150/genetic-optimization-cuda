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

        int maxId = 0;
        for (const auto &row : parser) {
            DataPoint<int, size> point;

            for (size_t i = 0; i < size; i++) {
                const auto cellValue = row[i];
                point.position[i] = std::stod(cellValue);
            }
            const std::string label = row[size];

            if (map.count(label) > 0) {
                point.label = map[label];
            } else {
                map[label] = ++maxId;
                point.label = maxId;
            }
            dataSet.push_back(point);
        }
    }

    std::string getLabelName(int label) {
        for (const auto &val : map) {
            if (val.second == label) {
                return val.first;
            }
        }
        throw std::runtime_error("label not found");
    }

  private:
    std::unordered_map<std::string, int> map;
};

template <int size>
thrust::host_vector<DataPoint<int, size>> loadDataset(std::string fileName) {

    return thrust::host_vector<DataPoint<int, size>>();
}
