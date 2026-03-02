#pragma once
#include <vector>
#include <cstdint>

class Statistics {
public:
    // Calcula y muestra por pantalla las estadísticas de todas las dimensiones
    static void computeAndPrint(const std::vector<float>& local_data, uint32_t cols, int rank, int size);
};