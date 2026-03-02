import numpy as np
import struct
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

# ==========================================
# CONFIGURACIÓN
# ==========================================
FILE_PATH = '../data/salida'
SAMPLE_SIZE = 4000  
N_CLUSTERS = 16     
N_NODES = 4         
MAX_ITER = 70       

NODE_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

def load_cpp_binary(filepath):
    print(f"Leyendo archivo binario: {filepath}...")
    with open(filepath, 'rb') as f:
        rows = struct.unpack('i', f.read(4))[0]
        cols = struct.unpack('i', f.read(4))[0]
        data = np.fromfile(f, dtype=np.float32)
        data = data.reshape((rows, cols))
        return data, cols

def main():
    try:
        data, cols = load_cpp_binary(FILE_PATH)
    except FileNotFoundError:
        print("¡Error! No se encuentra el archivo.")
        return

    print("Extrayendo muestra...")
    np.random.seed(42)
    indices = np.random.choice(data.shape[0], SAMPLE_SIZE, replace=False)
    data_sample = data[indices]

    # Si es 2D, no usamos PCA para que sea 100% fiel a la matemática visual
    if cols > 2:
        print("Aplicando PCA para reducir a 2D...")
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(data_sample)
    else:
        print("Dataset en 2D detectado. Saltando PCA...")
        data_2d = data_sample

    print("Preparando la simulación distribuida...")
    
    # EXPLOSIÓN INTELIGENTE: Elegimos puntos reales para asegurar que no haya centroides muertos
    initial_indices = np.random.choice(data_2d.shape[0], N_CLUSTERS, replace=False)
    real_centroids = data_2d[initial_indices]
    
    # Comprimimos su posición hacia el centro geométrico al 5% para el "Big Bang"
    mean_x, mean_y = np.mean(data_2d[:, 0]), np.mean(data_2d[:, 1])
    center = np.array([mean_x, mean_y])
    centroids = center + (real_centroids - center) * 0.05
    
    centroid_nodes = (np.arange(N_CLUSTERS) * N_NODES) // N_CLUSTERS
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    def update(frame):
        nonlocal centroids
        ax.clear()
        
        distances = np.linalg.norm(data_2d[:, np.newaxis] - centroids, axis=2)
        closest_clusters = np.argmin(distances, axis=1)
        point_nodes = (closest_clusters * N_NODES) // N_CLUSTERS
        p_colors = [NODE_COLORS[n] for n in point_nodes]
        
        ax.scatter(data_2d[:, 0], data_2d[:, 1], c=p_colors, s=12, alpha=0.4)
        
        c_colors = [NODE_COLORS[n] for n in centroid_nodes]
        ax.scatter(centroids[:, 0], centroids[:, 1], c=c_colors, marker='X', s=200, edgecolors='black', linewidths=2, zorder=5)
        
        ax.set_title(f'K-Means MPI Distribuido - Iteración {frame}\n(Puntos coloreados según la RAM del Nodo que los posee)', fontsize=12)
        ax.set_xlabel('Coordenada X')
        ax.set_ylabel('Coordenada Y')
        
        legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=NODE_COLORS[i], markersize=10, label=f'Nodo MPI {i}') for i in range(N_NODES)]
        ax.legend(handles=legend_elements, loc='upper right')
        
        new_centroids = np.array([data_2d[closest_clusters == k].mean(axis=0) if sum(closest_clusters == k) > 0 else centroids[k] for k in range(N_CLUSTERS)])
        
        if np.allclose(centroids, new_centroids):
            pass 
        else:
            centroids = new_centroids

    print("Generando el archivo GIF...")
    anim = FuncAnimation(fig, update, frames=MAX_ITER, interval=250, repeat_delay=3000)
    
    output_filename = 'mpi_kmeans_architecture.gif'
    anim.save(output_filename, writer='pillow', fps=5)
    print(f"¡Éxito! Animación guardada como {output_filename}")

if __name__ == '__main__':
    main()