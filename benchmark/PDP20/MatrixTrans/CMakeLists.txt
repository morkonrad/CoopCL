cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

project(MatrixTrans)

find_package(OpenCL REQUIRED)
if(NOT ${OpenCL_VERSION_MAJOR} MATCHES "2")
message(FATAL_ERROR "OpenCL 2.x not found")
endif(NOT ${OpenCL_VERSION_MAJOR} MATCHES "2")

find_package(OpenMP REQUIRED)

add_executable(${PROJECT_NAME} main.cpp)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 11)
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCL_LIBRARIES})
target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCL_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PRIVATE ../../../clDriver/lib/)
