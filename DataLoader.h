#pragma once
#include <vector>
#include <cstdint>

class DataLoader {
public:
    static void loadBinaryData (const char*filename, std :: vector <float> & data, uint32_t& rows, uint32_t& cols );
    static void printBinaryData(const std::vector<float>&data, const uint32_t rows, const uint32_t cols);
};