# High-Performance Distributed K-Means (C++)

![C++](https://img.shields.io/badge/C++-14%2B-blue.svg)
![OpenMPI](https://img.shields.io/badge/OpenMPI-Distributed_Memory-orange.svg)
![OpenMP](https://img.shields.io/badge/OpenMP-Shared_Memory-green.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)

Este proyecto es una implementación de **Altas Prestaciones (HPC)** del algoritmo de clustering K-Means. Diseñado para procesar conjuntos de datos masivos aprovechando arquitecturas de **memoria distribuida (OpenMPI)** y **memoria compartida (OpenMP)**.

## Rendimiento y Aceleración (Speedup)

El objetivo principal de este proyecto es exprimir al máximo el hardware subyacente. Partiendo de una implementación secuencial base, se han aplicado optimizaciones arquitectónicas severas:

* **Volumen de Datos:** Procesamiento de **8.000.000 de puntos** en **32 dimensiones** (Aprox. 1 GB de RAM en memoria continua).
* **Tiempo Secuencial Base:** ~81.68 segundos.
* **Tiempo Híbrido Optimizado:** **~2.74 segundos** (Media de 10 ejecuciones).
* **Aceleración Total (Speedup):** **~30x**.

### Técnicas de Optimización Aplicadas:
1. **Consciencia de Caché L1/L2:** Uso de vectores contiguos en memoria 1D (`std::vector<float>`) para evitar el *cache-miss* de los punteros dobles y maximizar el pre-fetching.
2. **Metaprogramación (Templates):** Desenrollado de bucles en tiempo de compilación (*Loop Unrolling*) para dimensiones fijas, permitiendo auto-vectorización SIMD profunda (`-O3 -march=native -ffast-math`).
3. **Paralelismo Bulk Synchronous (MPI):** Reducción de la latencia de red agrupando los puntos en "buzones" locales y utilizando comunicación colectiva (`MPI_Alltoallv`, `MPI_Allreduce`) una sola vez por iteración.
4. **Multihilo Seguro (OpenMP):** Uso de buzones privados por hilo (Thread-local storage) y bloques `#pragma omp critical` para evitar *Race Conditions* manteniendo la CPU al 100% de uso.

## Simulación de la Arquitectura Distribuida

![Animación K-Means Distribuido](docs/mpi_kmeans_architecture.gif)

> **Nota visual:** Esta simulación en 2D muestra el comportamiento real de la red. Los **4 colores** representan las memorias RAM físicas de los **4 Nodos MPI** distintos. A medida que los centroides se mueven, los puntos viajan por la red cambiando de dueño (función `MPI_Alltoallv`) hasta alcanzar el equilibrio matemático.

## Estructura del Proyecto

```text
📦 HPC-Distributed-KMeans
 ┣ 📂 src/          # Código fuente principal (K-Means distribuido, MPI, Estadísticas)
 ┣ 📂 include/      # Archivos de cabecera (.h)
 ┣ 📂 scripts/      # Utilidades en Python y scripts generadores de datos
 ┣ 📂 docs/         # Recursos gráficos y diagramas
 ┣ 📂 data/         # Carpeta ignorada en Git para datasets binarios
 ┣ 📜 Makefile      # Automatización de la compilación
 ┣ 📜 Dockerfile    # Contenedor para ejecución aislada y reproducible
 ┗ 📜 README.md

```
## Topología de Ejecución Híbrida

![Arquitectura Híbrida MPI+OpenMP](docs/hybrid_architecture.gif)

> **Flujo de Trabajo (BSP Model):** > 1. **(Azul) Distribución:** El dataset se reparte de forma equitativa entre los procesos aislados de OpenMPI (`MPI_Scatter`).
> 2. **(Verde) Procesamiento:** Cada nodo despierta a sus hilos físicos mediante OpenMP (`#pragma omp parallel`) para calcular distancias simultáneamente con memoria compartida local.
> 3. **(Rojo) Fusión:** Sincronización global y recolección de los datos optimizados (`MPI_Allreduce`).


## Reproducibilidad y Ejecución

Para garantizar que el código puede ser evaluado en cualquier sistema operativo sin lidiar con dependencias complejas de C++ y OpenMPI, el proyecto está completamente dockerizado.

### Opción A: Ejecución mediante Docker (Recomendada)
Requiere tener [Docker](https://www.docker.com/) instalado.

1. **Construir la imagen:**
   ```bash
   docker build -t hpc-kmeans .
   ```
2. **Ejecutar el clúster virtual (se autodestruye al terminar):**
   ```bash
   docker run --rm hpc-kmeans
   ```
   *Al ejecutar este comando, el contenedor generará automáticamente el dataset masivo y lanzará la simulación con 4 nodos MPI, mostrando las métricas y los cálculos estadísticos en la terminal.*

### Opción B: Compilación Nativa (Linux)
Si deseas ejecutarlo en tu propia máquina de forma nativa, requieres `g++` y `openmpi-bin`.

```bash
# 1. Compilar todo el ecosistema
make

# 2. Generar el dataset masivo en la carpeta data/
./GeneradorPuntosNDimensiones

# 3. Lanzar OpenMPI con 4 procesos paralelos aislados
mpirun --mca btl_vader_single_copy_mechanism none -np 4 ./main_mpi
```

## Cálculo Estadístico Distribuido (Map-Reduce)
Además del algoritmo K-Means, el sistema implementa un motor estadístico. Realiza dos pasadas óptimas por los datos segmentados en red para calcular **Mínimo, Máximo, Media y Varianza** de todas las dimensiones utilizando patrones de Map-Reduce (`MPI_Allreduce`) combinados con paralelismo híbrido OpenMP.