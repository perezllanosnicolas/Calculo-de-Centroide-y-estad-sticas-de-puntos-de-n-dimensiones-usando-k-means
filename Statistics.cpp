#include "Statistics.h"
#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <limits>
#include <iomanip>

void Statistics::computeAndPrint(const std::vector<float>& local_data, uint32_t cols, int rank, int size) {
    uint32_t local_rows = local_data.size() / cols;
    uint32_t total_rows = 0;
    
    // Obtener el número total de puntos en toda la red
    MPI_Allreduce(&local_rows, &total_rows, 1, MPI_UINT32_T, MPI_SUM, MPI_COMM_WORLD);

    if (total_rows == 0) return; // Seguridad

    // Vectores locales para almacenar los resultados por cada columna (dimensión)
    std::vector<float> local_min(cols, std::numeric_limits<float>::max());
    std::vector<float> local_max(cols, std::numeric_limits<float>::lowest());
    std::vector<double> local_sum(cols, 0.0); // Usamos double para evitar desbordamientos en la suma

    // ==========================================
    // PASADA 1: MIN, MAX y SUMA (OpenMP)
    // ==========================================
    #pragma omp parallel
    {
        std::vector<float> thread_min(cols, std::numeric_limits<float>::max());
        std::vector<float> thread_max(cols, std::numeric_limits<float>::lowest());
        std::vector<double> thread_sum(cols, 0.0);

        #pragma omp for nowait
        for (uint32_t i = 0; i < local_rows; i++) {
            for (uint32_t k = 0; k < cols; k++) {
                float val = local_data[i * cols + k];
                
                // Actualizar métricas privadas del hilo
                if (val < thread_min[k]) thread_min[k] = val;
                if (val > thread_max[k]) thread_max[k] = val;
                thread_sum[k] += val;
            }
        }

        #pragma omp critical
        {
            // Consolidar resultados al vector principal del nodo
            for (uint32_t k = 0; k < cols; k++) {
                if (thread_min[k] < local_min[k]) local_min[k] = thread_min[k];
                if (thread_max[k] > local_max[k]) local_max[k] = thread_max[k];
                local_sum[k] += thread_sum[k];
            }
        }
    }

    // ==========================================
    // FUSIÓN MPI 1
    // ==========================================
    std::vector<float> global_min(cols);
    std::vector<float> global_max(cols);
    std::vector<double> global_sum(cols);

    MPI_Allreduce(local_min.data(), global_min.data(), cols, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(local_max.data(), global_max.data(), cols, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(local_sum.data(), global_sum.data(), cols, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    std::vector<double> global_mean(cols);
    for (uint32_t k = 0; k < cols; k++) {
        global_mean[k] = global_sum[k] / total_rows;
    }

    // ==========================================
    // PASADA 2: VARIANZA (OpenMP)
    // ==========================================
    std::vector<double> local_variance_sum(cols, 0.0);

    #pragma omp parallel
    {
        std::vector<double> thread_variance_sum(cols, 0.0);

        #pragma omp for nowait
        for (uint32_t i = 0; i < local_rows; i++) {
            for (uint32_t k = 0; k < cols; k++) {
                float val = local_data[i * cols + k];
                double diff = val - global_mean[k];
                thread_variance_sum[k] += diff * diff;
            }
        }

        #pragma omp critical
        {
            for (uint32_t k = 0; k < cols; k++) {
                local_variance_sum[k] += thread_variance_sum[k];
            }
        }
    }

    // ==========================================
    // FUSIÓN MPI 2
    // ==========================================
    std::vector<double> global_variance_sum(cols);
    MPI_Allreduce(local_variance_sum.data(), global_variance_sum.data(), cols, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    std::vector<double> global_variance(cols);
    for (uint32_t k = 0; k < cols; k++) {
        global_variance[k] = global_variance_sum[k] / total_rows;
    }

    // ==========================================
    // IMPRIMIR RESULTADOS (Solo Nodo 0)
    // ==========================================
    if (rank == 0) {
        std::cout << "\n------------------------------------------------\n";
        std::cout << "ESTADÍSTICAS GLOBALES POR DIMENSIÓN\n";
        std::cout << "------------------------------------------------\n";
        
        // Imprimir las primeras dimensiones (ej. x, y, z o hasta 32 si quieres ver todo)
        //int cols_to_print = (cols > 5) ? 5 : cols; 
        int cols_to_print= cols; // Imprimir todas las dimensiones sin límite


        for (int k = 0; k < cols_to_print; k++) {
            std::cout << "Dimensión " << k << " (";
            if (k == 0) std::cout << "X";
            else if (k == 1) std::cout << "Y";
            else if (k == 2) std::cout << "Z";
            else std::cout << "Dim" << k;
            std::cout << "):" << std::endl;
            
            std::cout << "  - Mínimo  : " << std::fixed << std::setprecision(4) << global_min[k] << std::endl;
            std::cout << "  - Máximo  : " << global_max[k] << std::endl;
            std::cout << "  - Media   : " << global_mean[k] << std::endl;
            std::cout << "  - Varianza: " << global_variance[k] << std::endl;
        }
/* 
        if (cols > 5) {
            std::cout << "\n  (Y así hasta la dimensión " << cols << "...)\n";
        }
*/
        std::cout << "------------------------------------------------\n\n";
    }
}