#include <time.h>
#include <iostream>

#include "DataLoader.h"
#include "Statistics.h"
#include "KMeans.h"

int main()
{
    std::vector<float> data;
    uint32_t rows, cols;
    
    std::cout << "Cargando datos binarios..." << std::endl;
    DataLoader::loadBinaryData("salida", data, rows, cols);
    
    // Comentamos la impresión para que no tarde una eternidad con archivos grandes
    // DataLoader::printBinaryData(data, rows, cols); 
    
    // Aplicar Estadísticas
    /*
    clock_t start_stats = clock();
    float min = Statistics::getMin(data, rows, cols);
    float max = Statistics::getMax(data, rows, cols);
    float mean = Statistics::getMean(data, rows, cols);
    float variance = Statistics::getVariance(data, rows, cols);
    clock_t end_stats = clock();
    double stats_time = double(end_stats - start_stats) / double(CLOCKS_PER_SEC);
    std::cout << "Estadisticas calculadas en: " << stats_time << " segundos." << std::endl;
    */

    // Configurar KMeans
    uint32_t k = 32; // Ajustado a 32 clusters para que coincida con nuestro generador masivo
    KMeans kmeans(k);
    std::vector<float> centroids(k * cols);
    std::vector<uint32_t> assignments(rows);
    
    // Bucle para medir tiempo medio
    uint32_t nIteraciones = 10; // Puedes subirlo si es muy rápido, o bajarlo a 5 si tarda mucho
    std::cout << "Ejecutando K-Means " << nIteraciones << " veces para calcular la media de tiempo..." << std::endl;
    
    clock_t start = clock();
    
    for(uint32_t iter = 0; iter < nIteraciones; iter++) {
        // En cada iteración pasamos los mismos vectores. 
        // Como dentro de run() hacemos std::fill a 0, no importa que se reutilicen.
        kmeans.run(data, rows, cols, centroids, assignments);
    }
    
    clock_t end = clock();
    
    double time_taken_total = double(end - start) / double(CLOCKS_PER_SEC);
    double time_taken_avg = time_taken_total / nIteraciones;

    // Imprimir los resultados de la última iteración
    /*
    std::cout << "Centroids: " << std::endl;
    for (uint32_t i = 0; i < k; i++) {
        std::cout << "Cluster " << i << ": ";
        for (uint32_t j = 0; j < cols; j++) {
            std::cout << centroids[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
    */
    
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Tiempo TOTAL (" << nIteraciones << " execs): " << time_taken_total << " segundos" << std::endl;
    std::cout << "Tiempo MEDIO por exec: " << time_taken_avg << " segundos" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    return 0;
}