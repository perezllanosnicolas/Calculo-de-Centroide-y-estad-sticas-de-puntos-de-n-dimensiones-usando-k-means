#include "DataLoader.h"
#include <fstream>
#include <iostream>


    void DataLoader:: loadBinaryData (const char*filename, std :: vector <float> & data, uint32_t& rows, uint32_t& cols ){
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            std::cerr << "Error opening file: " << filename << std::endl;
            return;
        }
        // Read the number of rows and columns
        file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
        file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
        // Resize the data vector to hold all the values
        data.resize(rows * cols);
        // Read the binary data into the vector
        file.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
        file.close();
    }

    void DataLoader::printBinaryData(const std::vector<float>&data, const uint32_t rows, const uint32_t cols) {
        for (uint32_t i = 0; i < rows; ++i) {
            for (uint32_t j = 0; j < cols; ++j) {
                std::cout << data[i * cols + j] << " ";
            }
            std::cout << std::endl;
        }
    }

