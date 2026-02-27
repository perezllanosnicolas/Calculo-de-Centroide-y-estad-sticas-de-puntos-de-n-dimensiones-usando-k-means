#pragma once
#include <vector>
#include <cstdint>

class KMeans {
    private:
        uint32_t nClusters;
        uint32_t maxIterations = 2000;

   
        float calculateDistance(const float* p1, const float* p2, uint32_t cols);
    
    public:
        KMeans (uint32_t k);

        //Fución principal que ejecutará el bucle. 
        //Devolverá los centroides finales y las asignaciones.
        void run( const std::vector<float>& data, uint32_t rows, uint32_t cols, std::vector<float>& centroids, std::vector<uint32_t>& assignments);

};
