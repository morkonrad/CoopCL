cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

project(MatrixMul)

find_package(OpenCL REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(${PROJECT_NAME} main.cpp)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 11)
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCL_LIBRARIES})
target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCL_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PRIVATE ../../../clDriver/lib/)
