cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

project(Sift)

find_package(OpenCL REQUIRED)
if(NOT ${OpenCL_VERSION_MAJOR} MATCHES "2")
message(FATAL_ERROR "OpenCL 2.x not found")
endif(NOT ${OpenCL_VERSION_MAJOR} MATCHES "2")

if (UNIX)
find_package(Threads REQUIRED)
find_package(X11 REQUIRED)
endif (UNIX)

add_executable(${PROJECT_NAME} main.cpp)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 14)
target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCL_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PRIVATE ../../../clDriver/lib/)
target_include_directories(${PROJECT_NAME} PRIVATE ../../CImg)
target_link_libraries(${PROJECT_NAME} ${OpenCL_LIBRARIES})


if (UNIX)
target_link_libraries(${PROJECT_NAME} pthread)
target_link_libraries(${PROJECT_NAME} ${X11_LIBRARIES})
endif (UNIX)

if(MSVC)
  target_compile_options(${PROJECT_NAME} PRIVATE /W4)#/WX)
else(MSVC)
  target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -pedantic)#-Werror)
endif(MSVC)
