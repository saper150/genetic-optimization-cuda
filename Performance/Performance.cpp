#include "./Performance.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>

float average(const std::vector<float> &f) {
    return std::accumulate(f.begin(), f.end(), 0.f) / f.size();
}

float mediana(std::vector<float> &f) {
    std::nth_element(f.begin(), f.begin() + f.size() / 2, f.end());
    return f[f.size() / 2];
}

std::unordered_map<std::string, std::vector<float>> Performance::mesurements =
    std::unordered_map<std::string, std::vector<float>>();

using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::microseconds;

void Performance::mesure(std::string key, std::function<void()> function) {
    const high_resolution_clock::time_point t1 = high_resolution_clock::now();
    function();
    const high_resolution_clock::time_point t2 = high_resolution_clock::now();
    const auto dur = t2 - t1;
    const auto m = duration_cast<microseconds>(dur);

    mesurements[key].push_back(m.count() / 1000);
}

void Performance::print() {

    for (auto &&entry : mesurements) {
        std::cout << "-----------" << '\n';
        std::cout << entry.first << '\n';
        std::cout << "count: " << entry.second.size() << '\n';
        std::cout << "average: " << average(entry.second) << '\n';
        std::cout << "mediana: " << mediana(entry.second) << '\n';
    }
}
