#include <iostream>
#include <vector>

#include <cmath> // cos, sin 
#include <cstdlib> // rand, RAND_MAX 
#include <cstdio> // FILE, fopen_s, fwrite, fclose

#define PI 3.141582f

struct point2D {
    float x;
    float y;
};

point2D getRandomPoint(float x0, float y0,float maxRadius, float minRadius = 0.0f )
{
    point2D p;
    float r = minRadius + (maxRadius-minRadius) * (float)rand() / RAND_MAX;
    float alpha = 2.0f*PI* (float)rand() / RAND_MAX;
    p.x = x0 + r * cos(alpha);
    p.y = y0 + r * sin(alpha);
    return p;
};

int main()
{
    int nClusters = 3;
    int nPointsPerCluster = 50;

    std::vector<point2D> data;
    for (int i = 0; i < nClusters; i++)
    {
        point2D centroid = getRandomPoint(0.0f, 0.0f, 20.0, 0.0);
        for (int j = 0; j < nPointsPerCluster; j++)
            data.push_back(getRandomPoint(centroid.x,centroid.y, 1.0f));
    }
    FILE* resultsFile;
    resultsFile = fopen("salida", "wb");
    int nFilas = nClusters * nPointsPerCluster;
    int nCol = 2;
    fwrite(&nFilas, sizeof(int), 1, resultsFile);
    fwrite(&nCol, sizeof(int), 1, resultsFile);
    fwrite(data.data(), sizeof(float), data.size()*nCol, resultsFile);
    fclose(resultsFile);
    for (int i = 0; i < data.size(); i++)
        std::cout << data[i].x << "\t" << data[i].y << "\n";
}