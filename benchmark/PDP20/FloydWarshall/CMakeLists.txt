cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

project(FloydWarshall)

find_package(OpenCL REQUIRED)
add_executable(${PROJECT_NAME} main.cpp)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 11)
target_link_libraries(${PROJECT_NAME} ${OpenCL_LIBRARIES})
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCL_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PRIVATE ../../../clDriver/lib/)
