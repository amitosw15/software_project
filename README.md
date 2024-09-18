# KMeans and Spectral Clustering Project

This repository contains an implementation of the KMeans and Spectral Clustering algorithms in C, with Python bindings to allow usage within Python scripts. The project is designed to offer efficient clustering of large datasets and includes a precompiled executable for direct usage.

## Features

- **KMeans Clustering**: A popular algorithm for clustering data into a specified number of clusters.
- **Spectral Clustering**: A graph-based clustering algorithm that uses eigenvectors to determine cluster assignments.
- **Python Bindings**: The C code is accessible through Python, providing both performance and ease of use.
- **Optimized Performance**: High-performance code implemented in C, including matrix operations and eigenvalue calculations.
- **Precompiled Executable**: For users who prefer not to compile the code themselves, a precompiled executable is provided.

## File Structure

- **`spkmeans.c`**: Core C file containing the implementation of KMeans and Spectral Clustering algorithms.
- **`spkmeans.h`**: Header file for `spkmeans.c`, containing function declarations and macros used throughout the C code.
- **`kmeans.h`**: Header file focused on KMeans-specific functionality.
- **`kmeans2.h`**: Header file that might include variations or extensions to the KMeans algorithm.
- **`spkmeansmodule.c`**: C extension for binding the KMeans and Spectral Clustering functions to Python.
- **`spkmeans.py`**: Python wrapper script to interact with the compiled C extension.
- **`spkmeans.exe`**: Precompiled executable that allows running the clustering algorithms without compilation.
- **`setup.py`**: Python setup file for compiling and installing the C extension module.

## Installation

### Prerequisites

To compile and use this project, you will need:

- A C compiler (e.g., GCC or Clang)
- Python 3.x
- `setuptools` or `distutils` for building the Python extension

### Build Instructions

1. Clone the repository to your local machine:

   ```bash
   git clone <repository_url>
   cd <repository_directory>
