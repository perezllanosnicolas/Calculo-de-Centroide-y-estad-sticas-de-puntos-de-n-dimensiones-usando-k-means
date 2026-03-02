#include <mpi.h>
#include <iostream>
#include <vector>

#include "DataLoader.h"
 #include "Statistics.h"
#include "KMeans.h"

int main(int argc, char** argv)
{
    // Inicializar MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &size); 

    std::vector<float> data_global; 
    uint32_t rows = 0, cols = 0;

    // SOLO EL NODO 0 LEE EL ARCHIVO
    if (rank == 0) {
        std::cout << "Nodo 0: Cargando datos binarios..." << std::endl;
        DataLoader::loadBinaryData("salida", data_global, rows, cols);
    }

    // El Nodo 0 avisa a los demás de las dimensiones
    MPI_Bcast(&rows, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    uint32_t local_rows = rows / size; 
    std::vector<float> local_data(local_rows * cols);

    if (rank == 0) {
        std::cout << "Repartiendo " << rows << " puntos (" << cols << " dimensiones) entre " << size << " nodos..." << std::endl;
    }

    // Reparto de datos (Scatter)
    float* send_buffer = (rank == 0) ? data_global.data() : nullptr;
    MPI_Scatter(send_buffer, local_rows * cols, MPI_FLOAT, 
                local_data.data(), local_rows * cols, MPI_FLOAT,  
                0, MPI_COMM_WORLD);

    // Aplicar Estadísticas Híbridas (MPI + OpenMP)
    if (rank == 0) {
        std::cout << "Calculando Estadísticas del Dataset..." << std::endl;
    }
    
    // Todos los nodos calculan su parte y la fusionan en red
    Statistics::computeAndPrint(local_data, cols, rank, size);

    // Preparar el algoritmo KMeans...



    // Preparar el algoritmo
    uint32_t k = 32; 
    KMeans kmeans(k);
    std::vector<float> centroids(k * cols); 
    std::vector<uint32_t> local_assignments(local_rows);

    uint32_t nIteraciones = 10; // Ejecutamos 5 veces para hacer la media
    
    if (rank == 0) {
        std::cout << "Ejecutando K-Means " << nIteraciones << " veces para calcular la media de tiempo..." << std::endl;
    }

    // ==========================================
    // CRONÓMETRO MPI: INICIO
    // ==========================================
    MPI_Barrier(MPI_COMM_WORLD); // Sincronizamos a todos los nodos en la línea de salida
    double start_time = MPI_Wtime();

    for(uint32_t iter = 0; iter < nIteraciones; iter++) {
        // En memoria distribuida, cada iteración de prueba necesita empezar con los datos frescos
        // (ya que los puntos migran por la red modificando los tamaños locales).
        // Sin embargo, para esta prueba de estrés y medir rendimiento, correr run() varias veces seguidas nos vale.
        kmeans.run(local_data, cols, centroids, local_assignments, rank, size);
    }

    MPI_Barrier(MPI_COMM_WORLD); // Sincronizamos en la línea de meta
    double end_time = MPI_Wtime();
    // ==========================================
    // CRONÓMETRO MPI: FIN
    // ==========================================

    // Cada nodo ha tardado su propio tiempo. Cogemos el peor caso (el más lento marca el tiempo final)
    double local_time = end_time - start_time;
    double max_time = 0.0;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // El Nodo 0 imprime las estadísticas
    if (rank == 0) {
        double avg_time = max_time / nIteraciones;
        std::cout << "\n------------------------------------------------" << std::endl;
        std::cout << "MÉTRICAS DE RENDIMIENTO (MPI " << size << " nodos):" << std::endl;
        std::cout << "Tiempo TOTAL (" << nIteraciones << " execs): " << max_time << " segundos" << std::endl;
        std::cout << "Tiempo MEDIO por exec : " << avg_time << " segundos" << std::endl;
        std::cout << "------------------------------------------------\n" << std::endl;
    }

    // Finalizar MPI
    MPI_Finalize();
    return 0;

/*
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
    

    // Configurar KMeans
    uint32_t k = 32; // Ajustado a 32 clusters para que coincida con nuestro generador masivo
    KMeans kmeans(k);
    std::vector<float> centroids(k * cols);
    std::vector<uint32_t> assignments(rows);
    
    // Bucle para medir tiempo medio
    uint32_t nIteraciones = 50; // Puedes subirlo si es muy rápido, o bajarlo a 5 si tarda mucho
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
    
    
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Tiempo TOTAL (" << nIteraciones << " execs): " << time_taken_total << " segundos" << std::endl;
    std::cout << "Tiempo MEDIO por exec: " << time_taken_avg << " segundos" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    return 0;

    */
}