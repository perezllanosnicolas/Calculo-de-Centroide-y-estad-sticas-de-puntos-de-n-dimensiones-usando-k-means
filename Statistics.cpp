#include "Statistics.h"
#include <algorithm>
#include <numeric>
#include <cmath>

float Statistics::getMin(const std::vector<float>& data, const uint32_t& rows, const uint32_t& cols) {
    return *std::min_element(data.begin(), data.end());
}

float Statistics::getMax(const std::vector<float>& data, const uint32_t& rows, const uint32_t& cols) {
    return *std::max_element(data.begin(), data.end());
}

float Statistics::getMean(const std::vector<float>& data, const uint32_t& rows, const uint32_t& cols) {
    float sum = std::accumulate(data.begin(), data.end(), 0.0f);
    return sum / (rows * cols);
}

float Statistics::getVariance(const std::vector<float>& data, const uint32_t& rows, const uint32_t& cols) {
    float mean = getMean(data, rows, cols);
    float variance = 0.0f;
    for (const auto& value : data) {
        variance += (value - mean) * (value - mean);
    }
    return variance / (rows * cols);
}