#pragma once

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

struct Performance {
  private:
    static std::unordered_map<std::string, std::vector<float>> mesurements;

  public:
    static void mesure(std::string key, std::function<void()> function);
    static void print();
};
