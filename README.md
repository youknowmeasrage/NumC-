# Numc++ - NumPy for C++

[![C++14](https://img.shields.io/badge/C%2B%2B-14-blue.svg)](https://isocpp.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A comprehensive C++ library that provides NumPy-like functionality for numerical computing with 200+ functions including array operations, mathematical functions, statistics, linear algebra, and more.

## 🌟 Features

- **200+ NumPy Functions**: Complete suite of array operations
- **High Performance**: Optimized with -O3 and native architecture
- **Easy to Use**: NumPy-like API that's intuitive to developers
- **Header Only**: Just include and use (or link the .cpp)
- **Cross-Platform**: Works on Windows, Linux, macOS

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Documentation](#documentation)
- [Features](#features)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## 🚀 Installation

### From GitHub

```bash
# Clone the repository
git clone https://github.com/your-username/numcpp.git
cd numcpp/Numc++
```

### Build Example

**Linux/macOS:**
```bash
make
./example
```

**Windows:**
```bash
g++ -std=c++14 -c Numc++.cpp -o Numc++.o
g++ -std=c++14 -c example.cpp -o example.o
g++ -std=c++14 Numc++.o example.o -o example.exe
example.exe
```

### Use in Your Project

Copy `Numc++.h` and `Numc++.cpp` to your project and include:

```cpp
#include "Numc++.h"
using namespace numc;
```

Compile with: `g++ -std=c++14 -O3 your_code.cpp Numc++.cpp -o your_app`

## ⚡ Quick Start

```cpp
#include "Numc++.h"
using namespace numc;

int main() {
    // Create arrays
    Array ones = Array::ones({3, 3});
    Array zeros = Array::zeros({10});
    Array range = Array::arange(0, 10, 1);
    
    // Operations
    Array result = ones + 2.5;        // Scalar operations
    Array multiplied = ones * zeros;   // Element-wise
    
    // Statistics
    double mean_val = range.mean();
    double max_val = range.max();
    
    // Matrix operations
    Array I = Array::identity(3);
    Array transposed = ones.transpose();
    
    // Print
    result.print();
    
    return 0;
}
```

## 📚 Documentation

### Complete Documentation

- **[Function Reference](README.txt)** - Full list of 200+ functions
- **[Usage Guide](USAGE_GUIDE.txt)** - Comprehensive usage examples
- **[Installation Guide](INSTALLATION.md)** - Detailed installation instructions

### Key Function Categories

#### Array Creation
- `zeros()`, `ones()`, `full()` - Create arrays with initial values
- `arange()`, `linspace()`, `logspace()` - Create sequences
- `identity()`, `eye()` - Create special matrices

#### Mathematical Operations
- **Trigonometric**: `sin()`, `cos()`, `tan()`, `asin()`, `acos()`, `atan()`
- **Hyperbolic**: `sinh()`, `cosh()`, `tanh()`, `asinh()`, `acosh()`, `atanh()`
- **Exponential/Log**: `exp()`, `log()`, `log2()`, `log10()`, `sqrt()`, `cbrt()`
- **Rounding**: `floor()`, `ceil()`, `round()`, `trunc()`

#### Statistical Functions
- `sum()`, `mean()`, `std_dev()`, `var()` - Basic statistics
- `min()`, `max()`, `median()`, `percentile()` - Find extremes
- `argmax()`, `argmin()` - Find indices
- `unique()`, `unique_with_counts()` - Unique elements

#### Linear Algebra
- `dot()`, `matmul()` - Matrix multiplication
- `transpose()`, `inv()`, `pinv()` - Matrix operations
- `det()`, `trace()`, `rank()` - Matrix properties
- `diag()`, `triu()`, `tril()` - Extract parts

#### Array Manipulation
- `reshape()`, `transpose()`, `flatten()` - Change shape
- `concatenate()`, `stack()`, `vstack()`, `hstack()` - Combine arrays
- `split()`, `vsplit()`, `hsplit()` - Split arrays
- `repeat()`, `tile()` - Repeat elements
- `pad()`, `clip()` - Modify arrays

#### Advanced Features
- **Signal Processing**: `convolve()`, `autocorr()`, `xcorr()`
- **Interpolation**: `interp()`, `trapz()`, `cumtrapz()`
- **Normalization**: `normalize()`, `standardize()`
- **Sorting**: `sort()`, `argsort()`
- **Indexing**: `take()`, `put()`, `where()`

## 🎯 Examples

### Matrix Operations

```cpp
// Matrix multiplication
Array A = {{1, 2}, {3, 4}};
Array B = {{5, 6}, {7, 8}};
Array C = A.dot(B);

// Inverse
Array A_inv = A.inv();

// Determinant
double det = A.det();

// Eigenvalues and SVD
auto [eigenvals, eigenvecs] = A.eig();
auto [U, S, V] = A.svd();
```

### Statistical Analysis

```cpp
// Create data
Array data = Array::random_normal(0, 1, {1000});
data.fill(Array::random_normal(0, 1).get_data_ref().data());

// Compute statistics
double mean = data.mean();
double std = data.std_dev();
double median_val = data.median();
double percentile_95 = data.percentile(0.95);

// Find unique values
auto [unique_vals, counts] = data.unique_with_counts();
```

### Signal Processing

```cpp
// Generate signal
Array t = Array::linspace(0, 10, 1000);
Array signal = Array::sin(t);

// Convolution
Array kernel = Array::ones({10});
Array convolved = signal.convolve(kernel);

// Moving average
Array smoothed = signal.moving_average(20);

// Integration
double area = trapz(signal, t);
```

### Image Processing Simulation

```cpp
// Create image-like array
Array image = Array::random_uniform(0, 255, {256, 256});

// Apply transformations
Array normalized = image.normalize(0, 1);
Array standardized = image.standardize();

// Transpose (rotate)
Array rotated = image.transpose();

// Flip
Array flipped = image.flip();
```

## 📊 Performance

Numc++ is optimized for performance:

- **Compiled with -O3** for maximum optimization
- **Native architecture** support with `-march=native`
- **Efficient memory** layout with contiguous storage
- **Zero overhead** abstractions where possible

## 🛠️ Requirements

- **C++14** or later compiler (g++, clang++, MSVC 2017+)
- **Standard Library** only - no external dependencies
- **CMake** or Make (optional, for convenience)

## 🐛 Known Issues

- Some advanced functions (SVD, eigendecomposition) have simplified implementations
- Complex number support is limited
- Bitwise operations use integer approximations
- For issues or improvements, see [Contact](#contact)

## 👥 Contributing

Contributions are welcome! Steps:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

See [DISCLAIMER.txt](DISCLAIMER.txt) for details about authorship and AI-generated code.

## 📝 Disclaimer

This library was created with AI assistance. Only ~20 core functions were manually written, the remaining 200+ functions were generated through AI prompts. If you encounter errors, please report them:

**Email**: sabhay@zohomail.in

For full disclaimer, see [DISCLAIMER.txt](DISCLAIMER.txt)

## 📄 License

This project is released under the MIT License. See LICENSE file for details.

## 👤 Contact

For questions, bug reports, or contributions:

- **Email**: sabhay@zohomail.in
- **Issues**: Report on GitHub Issues
- **Documentation**: See README.txt and USAGE_GUIDE.txt

## 🙏 Acknowledgments

Inspired by NumPy for Python. This library brings NumPy's powerful array operations to C++ for high-performance numerical computing.

---

**Made with ❤️ for the C++ Community**

