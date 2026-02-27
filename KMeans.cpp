#include "KMeans.h"
#include <cmath> // sqrt
#include <iostream>

float KMeans::calculateDistance(const float* p1, const float* p2, uint32_t cols) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < cols; i++) {
        float diff = p1[i] - p2[i];
        sum += diff * diff;
    }
    return sum; // TRUCO CAP: No hace falta hacer sqrt() para comparar cuál es menor. ¡Te ahorras mucho tiempo de CPU!
}

// Constructor que inicializa el número de clusters
KMeans::KMeans (uint32_t k) : nClusters(k){}




void KMeans::run(const std::vector<float>& data, uint32_t rows, uint32_t cols, std::vector<float>& centroids, std::vector<uint32_t>& assignments){
    
    //Inciar el problema con una asignación incial de los puntos a los clusters 
    std::vector<uint32_t> counts(nClusters, 0);
    std::fill(centroids.begin(), centroids.end(), 0.0f);

    for (uint32_t i = 0; i < rows; i++) {
        // Un reparto equitativo simple: el punto 0 al grupo 0, el 1 al 1, etc.
        uint32_t cluster = i % nClusters; 
        assignments[i] = cluster;
        counts[cluster]++;
        for(uint32_t k= 0; k<cols; k++){
            centroids[cluster * cols + k] += data[i * cols + k];
        }
    }

    //Calcular los centroides iniciales basados en la asignación inicial
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
        // Asignar cada punto al cluster más cercano
        uint32_t puntosDesplazados = 0; // Contador de cambios de asignación 5%
        std:: fill(counts.begin(), counts.end(), 0); // Reiniciar los contadores para la nueva iteración
        std::fill(newCentroids.begin(), newCentroids.end(), 0.0f);

        for (uint32_t i = 0; i < rows; i++) {
            float minDist = std::numeric_limits<float>::max();
            uint32_t closestCluster = 0;
            for (uint32_t j = 0; j < nClusters; j++) {
                float dist = calculateDistance(&data[i * cols], &centroids[j * cols], cols); //Dado un punto y un centroide, calcula la distancia entre el punto y el centroide.
                if (dist < minDist) {
                    minDist = dist;
                    closestCluster = j;
                }
            }
            //Comprobar si el punto ha cambiado de cluster. Si es así, incrementamos el contador de puntos desplazados.
            if(assignments[i] != closestCluster) {
                puntosDesplazados++;
            }
            assignments[i] = closestCluster;
            counts[closestCluster]++;

            for (uint32_t k = 0; k < cols; k++) {
                newCentroids[closestCluster * cols + k] += data[i * cols + k];
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
        //Criterio de parada: Menor al 5%
        /*
        if(puntosDesplazados < (0.05 * rows)) {
            std::cout << "Convergencia alcanzada en la iteración " << iter << " con " << puntosDesplazados << " puntos desplazados." << std::endl;
            break;
        }
        */

    }


}

