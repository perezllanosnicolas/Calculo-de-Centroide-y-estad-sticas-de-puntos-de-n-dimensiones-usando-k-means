# ==========================================
# Makefile para HPC-Distributed-KMeans
# ==========================================

# Compilador y Flags de Altas Prestaciones (HPC)
CXX = mpic++
CXXFLAGS = -O3 -march=native -ffast-math -fopenmp -Iinclude -Wall

# Archivos y ejecutables
SRCS = src/main.cpp src/DataLoader.cpp src/KMeans.cpp src/Statistics.cpp
TARGET = main_mpi
GENERATOR_SRC = scripts/GeneradorPuntosNDimensiones.cpp
GENERATOR_TARGET = GeneradorPuntosNDimensiones

# Regla por defecto (compila todo)
all: $(TARGET) $(GENERATOR_TARGET)

# 1. Compilar el programa principal (K-Means distribuido)
# Al pasarle todos los .cpp de golpe, ayudamos al compilador a hacer inlining global
$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET)

# 2. Compilar el script generador de datos (Usa g++ normal porque no es MPI)
$(GENERATOR_TARGET): $(GENERATOR_SRC)
	g++ -O3 $(GENERATOR_SRC) -o $(GENERATOR_TARGET)

# Regla para limpiar los binarios generados
clean:
	rm -f $(TARGET) $(GENERATOR_TARGET)

.PHONY: all clean