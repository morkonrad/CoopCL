cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

project(vector_Add)

find_package(OpenCL REQUIRED)

add_executable(${PROJECT_NAME} main.cpp)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 11)
target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCL_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PRIVATE ../../../clDriver/lib/)
target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCL_LIBRARIES})