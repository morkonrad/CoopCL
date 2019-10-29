# CoopCL

What is this ? 
--------------


Requierments ?
---------------
1. C++14 compiler
2. CMake 3.x
3. OpenCL 2.x headers and lib, support for CPU and GPU
3. Graphic driver with OpenCL and SVM_FINE_GRAIN_BUFFER support
4. For unit-tests Ctest

How to build ?
---------------
  1. git clone /dst
  2. cd /dst
  3. mkdir build
  4. cd build
  5. cmake -G"Visual Studio 14 2015 Win64" .. 
  6. cmake --build . --config Release
  
For Windows, Visual Studio 2015 is a minimal tested version. For Linux it's tested with GCC 7.0 and clang 5.0. All compilers need to support C++14. 

Now after succesfull build you can call unit tests to check if they pass:  
 1. cd /clDriver
 2. ctest 
  
How to use it ?
----------------
After successful build and tests, the CoopCL should be ready to go.

It's header only library so yo need to onyl linkit whith your app.

Check sample usage/application below:




