#include <time.h>
#include <iostream>

#include "DataLoader.h"
#include "Statistics.h"
#include "KMeans.h"

int main()
{
    std::vector<float> data;
    uint32_t rows, cols;
    DataLoader::loadBinaryData("salida", data, rows, cols);
    DataLoader::printBinaryData(data, rows, cols);
    clock_t start, end;
    start = clock();
    float min = Statistics::getMin(data, rows, cols);
    float max = Statistics::getMax(data, rows, cols);
    float mean = Statistics::getMean(data, rows, cols);
    float variance = Statistics::getVariance(data, rows, cols);
    end = clock();
    double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
/*
    std::cout << "Min: " << min << std::endl;
    std::cout << "Max: " << max << std::endl;
    std::cout << "Mean: " << mean << std::endl;
    std::cout << "Variance: " << variance << std::endl;
    std::cout << "Time taken by function: " << time_taken << " seconds" << std::endl;
*/
    //Aplicar KMeans
    uint32_t k = 8; // Número de clusters
    KMeans kmeans(k);
    std::vector<float> centroids (k * cols) ;
    std::vector<uint32_t> assignments (rows) ;
    start = clock();
    kmeans.run(data, rows, cols, centroids, assignments);
    
    end = clock();
    time_taken = double(end - start) / double(CLOCKS_PER_SEC);

    std::cout << "Centroids: " << std::endl;
    for (uint32_t i = 0; i < k; i++) {
        std::cout << "Cluster " << i << ": ";
        for (uint32_t j = 0; j < cols; j++) {
            std::cout << centroids[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "Time taken by KMeans: " << time_taken << " seconds" << std::endl;

    return 0;

}