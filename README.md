# Tesis Ibex Clustering

Este repositorio contiene una versión modificada de la biblioteca Ibex, extendida para incorporar una estrategia de reinicio basada en clustering dentro del optimizador global. Esta implementación se desarrolla en el marco de una tesis y tiene como objetivo mejorar la eficiencia de la optimización global al evitar el estancamiento en mínimos locales.

## Descripción General

La optimización global es un problema complejo y computacionalmente costoso. Los algoritmos de optimización basados en la bisección del espacio de búsqueda pueden estancarse en regiones no prometedoras. Este proyecto introduce una estrategia de reinicio que utiliza algoritmos de clustering para identificar las regiones más prometedoras y reanudar la búsqueda a partir de ellas, mejorando así la convergencia del optimizador.

## Características Principales

*   **Optimización Global:** Basado en la robusta biblioteca Ibex para la optimización global con aritmética de intervalos.
*   **Reinicio Basado en Clustering:** Implementación de una estrategia de reinicio que utiliza algoritmos de clustering para escapar de mínimos locales y mejorar la eficiencia de la búsqueda.
*   **Algoritmos de Clustering:** Soporte para los algoritmos de clustering K-Means y DBSCAN.
*   **Configuración Flexible:** Parámetros configurables para ajustar el comportamiento del clustering y de la estrategia de reinicio.

## Estructura del Repositorio

```
Tesis_ibex_clustering/
├── benchs/               # Ficheros de benchmark para pruebas
│   └── optim/
├── src/                  # Código fuente modificado de Ibex
│   └── optim/
│       ├── ibex_Optimizer.h
│       ├── ibex_Optimizer.cpp
│       └── ibex_DefaultOptimizerConfig.cpp
└── README.md            
```

## Instalación y Compilación

Para compilar el proyecto, es necesario tener la biblioteca Ibex instalada. Una vez clonado el repositorio, los ficheros modificados en `src/optim/` deben reemplazar a los ficheros originales en el directorio de Ibex.

1.  **Clonar el repositorio:**
    ```bash
    git clone https://github.com/Chelosky-O/Tesis_ibex_clustering.git
    ```

2.  **Reemplazar los ficheros en Ibex:**
    Copie los ficheros de `src/optim/` a la carpeta `src/optim/` de su instalación de Ibex.

3.  **Compilar Ibex:**
    Siga las instrucciones de compilación de la biblioteca Ibex, un ejemplo de ejecución abajo.
    ```
    ./ibexopt [ibex-lib-path]/benchs/optim/medium/ex2_1_7.bch 
    ```
    


## Uso

La nueva estrategia de reinicio se activa automáticamente cuando el optimizador detecta un estancamiento. La configuración de los parámetros del clustering se realiza en `Optimizer::Optimizer` del archivo `ibex_Optimizer.cpp`.

### Ejemplo de Configuración

A continuación se muestra un ejemplo de cómo configurar los parámetros de clustering:

```cpp
// Parámetros para reinicios basados en clustering

    clustering_params.choice = ClusteringParams::Algorithm::DBSCAN; // Algoritmos disponibles (DBSCAN o KMEANS)

    // KMEANS
    clustering_params.k = 10;     // Número de clusters (Si no esta activo el k dinamico se utiliza este)

    // DBSCAN
    clustering_params.eps = 0.1;   // Radio para DBSCAN
    clustering_params.minPts = this->n + 1;     // Tamaño mínimo de cluster para DBSCAN (numero de dimensiones del problema + 1)

    //EPS DINAMICO
    clustering_params.use_dynamic_eps = true; // Activar cálculo dinámico de eps
    clustering_params.kneedle_alpha = 0.5;  // Factor de suavizado para el cálculo de eps 0.5 POR DEFECTO

    //FILTRO DE VOLUMEN
    double hull_volume_threshold_fraction = 3.0; // Umbral para el volumen

    //ACTIVADORES DE REINICIO
	restart_threshold = 500; //Umbral inicial de numero de iteraciones sin mejora
	node_threshold = 50000; //Umbral inicial de numero de nodos antes de un reinicio

    //En Optimizer::Status Optimizer::optimize() se encuentra la configuración de como se penalizan los threshold de iteraciones sin mejora y numero de nodos maximo.
```

## Benchmarks

El directorio `benchs/` contiene una serie de problemas de benchmark para probar el rendimiento del optimizador, estos fueron los utilizados en la memoria. Para ejecutar los benchmarks, utilice el ejecutable de Ibex con los ficheros de benchmark proporcionados.
