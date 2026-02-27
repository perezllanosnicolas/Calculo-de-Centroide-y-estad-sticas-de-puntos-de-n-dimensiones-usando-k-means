#pragma once
#include <vector>
#include <cstdint>

class Statistics {
    public: 
        static float getMin(const std:: vector <float>& data, const uint32_t& rows, const uint32_t& cols);
        static float getMax(const std:: vector <float>& data, const uint32_t& rows, const uint32_t& cols);
        static float getMean(const std:: vector <float>& data, const uint32_t& rows, const uint32_t& cols);
        static float getVariance(const std:: vector <float>& data, const uint32_t& rows, const uint32_t& cols);
};