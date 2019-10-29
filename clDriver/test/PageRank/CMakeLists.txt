cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

project(test_page_rank)

find_package(OpenCL REQUIRED)
find_package(OpenMP REQUIRED)


add_executable(${PROJECT_NAME} main.cpp)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 11)
target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCL_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PRIVATE ../../lib)
target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCL_LIBRARIES})
target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
add_test(Test_page_rank ${PROJECT_NAME})
