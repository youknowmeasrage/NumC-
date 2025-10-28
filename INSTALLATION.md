# Numc++ - Installation Guide

## 1) Download from GitHub

- Using git:
  git clone https://github.com/your-user/numcpp.git
  cd numcpp/Numc++

- Without git:
  - Go to the repository page
  - Click Code → Download ZIP
  - Extract, then open the Numc++ folder

## 2) Build the example

- Linux/macOS (requires g++, make):
  make
  ./example

- Windows (MinGW g++):
  g++ -std=c++14 -O3 -c Numc++.cpp -o Numc++.o
  g++ -std=c++14 -O3 -c example.cpp -o example.o
  g++ -std=c++14 -O3 Numc++.o example.o -o example.exe
  example.exe

If your compiler supports C++23, replace -std=c++14 with -std=c++23.

## 3) Use in your project

Option A - Copy files:
- Copy Numc++.h and Numc++.cpp into your source tree
- Include in your code: #include "Numc++.h"
- Compile and link Numc++.cpp with your sources

Option B - Add as a submodule:
  git submodule add https://github.com/your-user/numcpp.git external/numcpp
  # Add external/numcpp/Numc++ to your include paths

CMake sample (Linux/macOS):
  add_library(numcpp STATIC
      ${CMAKE_CURRENT_SOURCE_DIR}/external/numcpp/Numc++/Numc++.cpp)
  target_include_directories(numcpp PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/external/numcpp/Numc++)
  target_compile_features(numcpp PUBLIC cxx_std_14)
  target_compile_options(numcpp PUBLIC -O3)

## 4) Troubleshooting

- g++: unrecognized option -std=c++23 → use -std=c++14
- 'make' not found on Windows → use the Windows commands above
- Link errors → ensure Numc++.cpp is compiled and linked with your app
- Performance → add -O3 -march=native for best speed

## 5) Contact

Report issues to: sabhay@zohomail.in
