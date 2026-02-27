#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cstdio>
#include <limits>

int main() {
    srand(time(NULL)); // Inicializar la semilla aleatoria real

    int nClusters = 32;
    int nPointsPerCluster = 250000; // ¡Un millón de puntos por cluster para un total de 16 millones de puntos!
    int nDimensions = 32; // ¡Ahora puedes poner las dimensiones que quieras!

    int nFilas = nClusters * nPointsPerCluster;
    int nCol = nDimensions;

    std::vector<float> data(nFilas * nCol);
    
    // 1. Generar los centroides base (coordenadas aleatorias entre -20 y 20)
    std::vector<float> centroids(nClusters * nCol);
    for(int i = 0; i < nClusters; i++) {
        for(int d = 0; d < nCol; d++) {
            centroids[i * nCol + d] = -20.0f + 40.0f * ((float)rand() / RAND_MAX); 
        }
    }

    // 2. Generar los puntos alrededor de sus centroides correspondientes
    for(int i = 0; i < nClusters; i++) {
        for(int j = 0; j < nPointsPerCluster; j++) {
            int pointIndex = i * nPointsPerCluster + j;
            for(int d = 0; d < nCol; d++) {
                // Desplazamiento aleatorio entre -2.0 y 2.0
                float offset = -2.0f + 4.0f * ((float)rand() / RAND_MAX); 
                data[pointIndex * nCol + d] = centroids[i * nCol + d] + offset;
            }
        }
    }

    // 3. MEZCLAR LOS PUNTOS (Fisher-Yates Shuffle)
    // Para que no queden ordenados por clúster en el archivo
    for (int i = nFilas - 1; i > 0; --i) {
        int j = rand() % (i + 1);
        // Intercambiar el punto 'i' con el punto 'j' (todas sus dimensiones)
        for(int d = 0; d < nCol; d++) {
            float temp = data[i * nCol + d];
            data[i * nCol + d] = data[j * nCol + d];
            data[j * nCol + d] = temp;
        }
    }

    // 4. Escribir al archivo binario
    FILE* resultsFile = fopen("salida", "wb");
    if(resultsFile) {
        fwrite(&nFilas, sizeof(int), 1, resultsFile);
        fwrite(&nCol, sizeof(int), 1, resultsFile);
        fwrite(data.data(), sizeof(float), data.size(), resultsFile);
        fclose(resultsFile);
        std::cout << "Generados " << nFilas << " puntos mezclados en " << nCol << " dimensiones." << std::endl;
    } else {
        std::cout << "Error al abrir el archivo." << std::endl;
    }

    return 0;
}