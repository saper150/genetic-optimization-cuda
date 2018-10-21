#pragma once

#include "../Knn/Knn.cuh"
#include <sstream>

template <int atributeCount, int labelsCount, int k> struct KnnFitness {

    Knn<atributeCount, labelsCount, k> knn;

    KnnFitness(thrust::host_vector<LabelDataPoint<atributeCount>> &data)
        : knn(data) {

        validateLabelsCount(data);

        

        // for (size_t i = 0; i < data.size(); i++) {
        //     for (size_t j = 0; j < data.size(); j++) {
        //         std::cout
        //             << std::setprecision(4)
        //             << ((Neabour)nearestNeabours[i * data.size() + j]).label
        //             << "|"
        //             << ((Neabour)nearestNeabours[i * data.size() + j]).index
        //             << "\t|";
        //     }
        //     std::cout << std::endl;
        // }
    }

    void validateLabelsCount(
        const thrust::host_vector<LabelDataPoint<atributeCount>> &data) {
        const LabelDataPoint<atributeCount> max = *(thrust::max_element(
            data.begin(), data.end(),
            [](LabelDataPoint<atributeCount> a,
               LabelDataPoint<atributeCount> b) { return a.label < b.label; }));

        if ((max.label + 1) != labelsCount) {
            std::cout << (max.label + 1) << "|" << labelsCount << "|"
                      << (int)((max.label + 1) != labelsCount) << "\n";
            std::ostringstream stringStream;
            stringStream
                << "labelsCount and actual labels does not match, expected "
                << labelsCount << " got: " << (max.label + 1);
            throw stringStream.str().c_str();
        }
    }

    void accuracy(Population<bool> &population,
                  thrust::device_vector<float> &accuracy) {
        return knn.accuracy(population, accuracy);
    }
    auto &operator()(Population<bool> &population,
                     thrust::device_vector<float> &fitness) {

        return fitness;
    }
};
