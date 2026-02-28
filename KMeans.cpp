#include "KMeans.h"
#include <cmath> 
#include <iostream>
#include <limits>
#include <algorithm> // para std::fill

#include <mpi.h> // Para usar MPI dentro de KMeans 

// 1. APLICACIÓN DE METAPROGRAMACIÓN (TEMA 1)
// Creamos una función plantilla (template) independiente. 
// Al pasar DIM entre < >, el compilador conoce el tamaño y elimina el bucle for.
template <uint32_t DIM>
inline float calcDistOptimized(const float* p1, const float* p2) {
    float sum = 0.0f;
    // El compilador desenrollará este bucle automáticamente gracias al template
    for (uint32_t i = 0; i < DIM; i++) {
        float diff = p1[i] - p2[i];
        sum += diff * diff;
    }
    return sum; 
}

// Tu función original actúa como "fallback" genérico por si la dimensión no es 32 ni 4
float KMeans::calculateDistance(const float* p1, const float* p2, uint32_t cols) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < cols; i++) {
        float diff = p1[i] - p2[i];
        sum += diff * diff;
    }
    return sum; 
}

// Constructor que inicializa el número de clusters
KMeans::KMeans (uint32_t k) : nClusters(k){}


void KMeans::run(std::vector<float>& local_data, uint32_t cols, std::vector<float>& centroids, std::vector<uint32_t>& assignments, int rank, int size) {
    
    std::vector<uint32_t> local_counts(nClusters, 0);
    std::vector<float> local_centroids_sum(nClusters * cols, 0.0f);
    std::fill(centroids.begin(), centroids.end(), 0.0f);

    // Obtener el total de puntos globales para el criterio de parada del 5%
    uint32_t local_rows = local_data.size() / cols;
    uint32_t total_rows = 0;
    MPI_Allreduce(&local_rows, &total_rows, 1, MPI_UINT32_T, MPI_SUM, MPI_COMM_WORLD);

    // Asignación inicial
    uint32_t global_offset = rank * local_rows;

    for (uint32_t i = 0; i < local_rows; i++) {
        uint32_t global_index = global_offset + i;
        uint32_t cluster = global_index % nClusters; 
        assignments[i] = cluster;
        local_counts[cluster]++;
        for(uint32_t k = 0; k < cols; k++){
            local_centroids_sum[cluster * cols + k] += local_data[i * cols + k];
        }
    }
    
    // Fusión en red (MPI_Allreduce) de la inicialización
    std::vector<uint32_t> global_counts(nClusters, 0);
    MPI_Allreduce(local_counts.data(), global_counts.data(), nClusters, MPI_UINT32_T, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(local_centroids_sum.data(), centroids.data(), nClusters * cols, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    
    for(uint32_t j=0; j<nClusters; j++){
        if(global_counts[j] > 0){
            for(uint32_t k=0; k<cols; k++){
                centroids[j * cols + k] /= global_counts[j];
            }
        }
    }

    // ------ BUCLE PRINCIPAL (MEMORIA DISTRIBUIDA) ------
    for (uint32_t iter = 0; iter < maxIterations; iter++) {
        uint32_t puntosDesplazados = 0; 
        uint32_t current_local_rows = local_data.size() / cols;

        // FASE 1: CREAR LOS BUZONES DE SALIDA
        std::vector<std::vector<float>> outgoing_mailboxes(size);
        std::vector<std::vector<uint32_t>> outgoing_assignments(size); // ¡NUEVO! Buzón para las etiquetas

        for (uint32_t i = 0; i < current_local_rows; i++) {
            float minDist = std::numeric_limits<float>::max();
            uint32_t closestCluster = 0;
            
            for (uint32_t j = 0; j < nClusters; j++) {
                float dist = 0.0f;
                if (cols == 32) {
                    dist = calcDistOptimized<32>(&local_data[i * cols], &centroids[j * cols]);
                } else {
                    dist = calculateDistance(&local_data[i * cols], &centroids[j * cols], cols); 
                }

                if (dist < minDist) {
                    minDist = dist;
                    closestCluster = j;
                }
            }
            
            // Comprobar si cambió de grupo
            if (i < assignments.size() && assignments[i] != closestCluster) {
                puntosDesplazados++;
            }

            // Calcular a qué nodo pertenece ahora
            int target_node = (closestCluster * size) / nClusters;

            // Guardar datos en el buzón correspondiente
            for (uint32_t k = 0; k < cols; k++) {
                outgoing_mailboxes[target_node].push_back(local_data[i * cols + k]);
            }
            outgoing_assignments[target_node].push_back(closestCluster);
        }

        // FASE 2: PREPARAR EL ENVÍO (Aplanar buzones para MPI_Alltoallv)
        std::vector<int> send_points(size, 0); // Puntos a enviar
        std::vector<int> send_floats(size, 0); // Floats a enviar
        std::vector<int> displs_points(size, 0);
        std::vector<int> displs_floats(size, 0);
        
        int total_send_points = 0;
        int total_send_floats = 0;

        for (int n = 0; n < size; n++) {
            send_points[n] = outgoing_assignments[n].size();
            send_floats[n] = outgoing_mailboxes[n].size();
            
            displs_points[n] = total_send_points;
            displs_floats[n] = total_send_floats;
            
            total_send_points += send_points[n];
            total_send_floats += send_floats[n];
        }

        // Aplanar los vectores
        std::vector<float> send_buffer_floats;
        send_buffer_floats.reserve(total_send_floats);
        std::vector<uint32_t> send_buffer_assigns;
        send_buffer_assigns.reserve(total_send_points);

        for (int n = 0; n < size; n++) {
            send_buffer_floats.insert(send_buffer_floats.end(), outgoing_mailboxes[n].begin(), outgoing_mailboxes[n].end());
            send_buffer_assigns.insert(send_buffer_assigns.end(), outgoing_assignments[n].begin(), outgoing_assignments[n].end());
        }

        // FASE 3: INTERCAMBIO EN RED
        // 3.1 Preguntar cuántos puntos me va a mandar cada nodo
        std::vector<int> recv_points(size, 0);
        MPI_Alltoall(send_points.data(), 1, MPI_INT, recv_points.data(), 1, MPI_INT, MPI_COMM_WORLD);

        std::vector<int> recv_floats(size, 0);
        std::vector<int> recv_displs_points(size, 0);
        std::vector<int> recv_displs_floats(size, 0);
        
        int total_recv_points = 0;
        int total_recv_floats = 0;
        for (int n = 0; n < size; n++) {
            recv_floats[n] = recv_points[n] * cols;
            recv_displs_points[n] = total_recv_points;
            recv_displs_floats[n] = total_recv_floats;
            total_recv_points += recv_points[n];
            total_recv_floats += recv_floats[n];
        }

        // 3.2 Intercambio masivo real (El Camión de Mudanzas)
        std::vector<float> new_local_data(total_recv_floats);
        MPI_Alltoallv(send_buffer_floats.data(), send_floats.data(), displs_floats.data(), MPI_FLOAT,
                      new_local_data.data(), recv_floats.data(), recv_displs_floats.data(), MPI_FLOAT, MPI_COMM_WORLD);

        std::vector<uint32_t> new_assignments(total_recv_points);
        MPI_Alltoallv(send_buffer_assigns.data(), send_points.data(), displs_points.data(), MPI_UINT32_T,
                      new_assignments.data(), recv_points.data(), recv_displs_points.data(), MPI_UINT32_T, MPI_COMM_WORLD);

        // ¡Actualizamos nuestros datos! A partir de aquí somos dueños de los nuevos puntos
        local_data = std::move(new_local_data);
        assignments = std::move(new_assignments);
        current_local_rows = local_data.size() / cols;

        // FASE 4: RECÁLCULO DE CENTROIDES
        std::fill(local_counts.begin(), local_counts.end(), 0);
        std::fill(local_centroids_sum.begin(), local_centroids_sum.end(), 0.0f);
        
        for (uint32_t i = 0; i < current_local_rows; i++) {
            uint32_t cluster = assignments[i];
            local_counts[cluster]++;
            for (uint32_t k = 0; k < cols; k++) {
                local_centroids_sum[cluster * cols + k] += local_data[i * cols + k];
            }
        }

        // Fusión global para construir los centroides oficiales
        std::fill(global_counts.begin(), global_counts.end(), 0);
        MPI_Allreduce(local_counts.data(), global_counts.data(), nClusters, MPI_UINT32_T, MPI_SUM, MPI_COMM_WORLD);

        std::vector<float> global_centroids_sum(nClusters * cols, 0.0f);
        MPI_Allreduce(local_centroids_sum.data(), global_centroids_sum.data(), nClusters * cols, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

        for (uint32_t j = 0; j < nClusters; j++) {
            if (global_counts[j] > 0) {
                for (uint32_t k = 0; k < cols; k++) {
                    centroids[j * cols + k] = global_centroids_sum[j * cols + k] / global_counts[j];
                }
            }
        }

        // FASE 5: CRITERIO DE PARADA (Según el PDF: "total desplazados respecto al total de datos")
        uint32_t global_desplazados = 0;
        MPI_Allreduce(&puntosDesplazados, &global_desplazados, 1, MPI_UINT32_T, MPI_SUM, MPI_COMM_WORLD);

        if (global_desplazados < (0.05f * total_rows)) {
            if (rank == 0) {
                //std::cout << "Convergencia en iteración " << iter 
                //          << " con " << global_desplazados << " desplazados." << std::endl;
            }
            break;
        }
    }
}