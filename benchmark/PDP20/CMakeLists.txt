cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

project(benchmark)
add_subdirectory(Mandelbrot)
add_subdirectory(MatrixMul)
add_subdirectory(MatrixTrans)
add_subdirectory(Sobel)
add_subdirectory(BlackScholes) 
add_subdirectory(PageRank)
add_subdirectory(FloydWarshall)
add_subdirectory(Sift)
add_subdirectory(vAdd)

