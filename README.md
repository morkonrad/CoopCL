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
  5. cmake -G"Visual Studio 14 2015 Win64" .. (Visual Studio 2015 is a minimal tested version, older version not tested yet)
  6. cmake --build . --config Release
  
  Now after succesfull build you can call unit tests to check if they pass:  
  cd /clDriver
  ctest 
  
How to use it ?
----------------
After successful build and tests, the CoopCL should be ready to go.

It's header only library so yo can link/use it whatever you want.

Check sample usage/application below:




