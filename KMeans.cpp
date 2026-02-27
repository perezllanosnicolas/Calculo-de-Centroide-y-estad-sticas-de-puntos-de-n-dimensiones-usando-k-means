#include "KMeans.h"
#include <cmath> 
#include <iostream>
#include <limits>
#include <algorithm> // para std::fill

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


void KMeans::run(const std::vector<float>& data, uint32_t rows, uint32_t cols, std::vector<float>& centroids, std::vector<uint32_t>& assignments){
    
    // Inciar el problema con una asignación incial de los puntos a los clusters 
    std::vector<uint32_t> counts(nClusters, 0);
    std::fill(centroids.begin(), centroids.end(), 0.0f);

    for (uint32_t i = 0; i < rows; i++) {
        uint32_t cluster = i % nClusters; 
        assignments[i] = cluster;
        counts[cluster]++;
        for(uint32_t k= 0; k<cols; k++){
            centroids[cluster * cols + k] += data[i * cols + k];
        }
    }

    // Calcular los centroides iniciales basados en la asignación inicial
    for (uint32_t j = 0; j < nClusters; j++) {
        if (counts[j] > 0) {
            for (uint32_t k = 0; k < cols; k++) {
                centroids[j * cols + k] /= counts[j];
            }
        }
    }

    // Bucle principal de KMeans
    std::vector<float> newCentroids(nClusters * cols, 0.0f);

    for (uint32_t iter = 0; iter < maxIterations; iter++) {
        uint32_t puntosDesplazados = 0; 
        std::fill(counts.begin(), counts.end(), 0); 
        std::fill(newCentroids.begin(), newCentroids.end(), 0.0f);

        for (uint32_t i = 0; i < rows; i++) {
            float minDist = std::numeric_limits<float>::max();
            uint32_t closestCluster = 0;
            
            for (uint32_t j = 0; j < nClusters; j++) {
                float dist = 0.0f;
                
                // 2. USO DE LA PLANTILLA SEGÚN LA DIMENSIÓN
                if (cols == 32) {
                    // Llamada super-optimizada en tiempo de compilación
                    dist = calcDistOptimized<32>(&data[i * cols], &centroids[j * cols]);
                } else if (cols == 4) {
                    // Llamada optimizada para 4 dimensiones
                    dist = calcDistOptimized<4>(&data[i * cols], &centroids[j * cols]);
                } else {
                    // Llamada estándar
                    dist = calculateDistance(&data[i * cols], &centroids[j * cols], cols); 
                }

                if (dist < minDist) {
                    minDist = dist;
                    closestCluster = j;
                }
            }
            
            if(assignments[i] != closestCluster) {
                puntosDesplazados++;
            }
            assignments[i] = closestCluster;
            counts[closestCluster]++;

            // 3. OPTIMIZACIÓN DEL PUNTERO DIRECTO PARA SUMAR (Evita multiplicar en cada iteración)
            float* targetCentroid = &newCentroids[closestCluster * cols];
            const float* currentPoint = &data[i * cols];
            
            // También podemos ayudar al compilador aquí con un if
            if (cols == 32) {
                for (uint32_t k = 0; k < 32; k++) targetCentroid[k] += currentPoint[k];
            } else {
                for (uint32_t k = 0; k < cols; k++) targetCentroid[k] += currentPoint[k];
            }
        }

        // Actualizar los centroides
        for (uint32_t j = 0; j < nClusters; j++) {
            if (counts[j] > 0) {
                for (uint32_t k = 0; k < cols; k++) {
                    centroids[j * cols + k] = newCentroids[j * cols + k] / counts[j];
                }
            }
        }
        
        // Criterio de parada: Menor al 5%
        if(puntosDesplazados < (0.05f * rows)) {
            //std::cout << "Convergencia alcanzada en la iteración " << iter << " con " << puntosDesplazados << " puntos desplazados." << std::endl;
            break;
        }
    }
}